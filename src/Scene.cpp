/* This file is part of the Marble Marcher (https://github.com/HackerPoet/MarbleMarcher).
* Copyright(C) 2018 CodeParade
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.If not, see <http://www.gnu.org/licenses/>.
*/
#include "Scene.h"
#include "Res.h"
#include <iostream>
#include <fractals/FractalInclude.hpp>

static const float PI = 3.14159265359f;
static const float ground_force = 0.008f;
static const float air_force = 0.004f;
static const float ground_friction = 0.99f;
static const float air_friction = 0.995f;
static const float orbit_speed = 0.005f;
static const int max_marches = 10;
static const int num_phys_steps = 6;
static const float marble_bounce = 1.2f; //Range 1.0 to 2.0
static const float orbit_smooth = 0.995f;
static const float zoom_smooth = 0.85f;
static const float look_smooth = 0.75f;
static const float look_smooth_free_camera = 0.9f;
static const int frame_transition = 400;
static const int frame_orbit = 600;
static const int frame_deorbit = 800;
static const int frame_countdown = frame_deorbit + 3*60;
static const float default_zoom = 15.0f;
static const float ground_ratio = 1.15f;
static const int mus_switches[num_level_music] = {9, 15, 21, 24};
static const int num_levels_midpoint = 15;

int cur_frame = 0;
bool recording = false;
bool replay = false;

std::vector<InputRecord> recording_data;
std::fstream input_recording;

int* GetReplayFrame()
{
	return &cur_frame;
}

void StartRecording()
{
	if (!replay)
	{
		recording_data.clear();
		recording = true;
	}
}

void StopRecording2File(std::string path, bool save)
{
	if (recording)
	{
		recording = false;
		if (save)
		{
			input_recording.open(path, std::ios::out | std::ios::binary | std::ios::trunc);

			if (!input_recording.is_open())
			{
				//DisplayError("Error opening record file");
			}
			else
			{
				for (auto &input : recording_data)
				{
					input_recording.write((char*)&input, sizeof(InputRecord));
				}
			}

			input_recording.close();
		}

		recording_data.clear();
	}
}

void StartReplayFromFile(std::string path)
{
	if (!recording)
	{
		recording_data.clear();
		replay = true;
		input_recording.open(path, std::ios::in | std::ios::binary);

		if (!input_recording.is_open())
		{
			replay = false;
			//DisplayError("Error opening record file");
		}
		else
		{
			input_recording.seekg(0, input_recording.beg);
			InputRecord rec;
			while (input_recording)
			{
				input_recording.read((char*)&rec, sizeof(InputRecord));
				recording_data.push_back(rec);
			}
			cur_frame = 0;
		}

		input_recording.close();
	}
}

void StopReplay()
{
	if (!recording)
	{
		replay = false;
		recording_data.clear();
	}
}


void SaveRecord(float mx, float my, float vx, float vy, float cz, bool mc)
{
	if (recording && !replay)
	{
		InputRecord rec;
		rec.move_x = mx;
		rec.move_y = my;
		rec.view_x = vx;
		rec.view_y = vy;
		rec.cam_z = cz;
		rec.mouse_clicked = mc;

		recording_data.push_back(rec);
	}
}

void AddResetPadding()
{
	for (int i = 0; i < frame_deorbit; i++)
	{
		SaveRecord(0, 0, 0, 0, 0, 0);
	}
}

InputRecord GetRecord()
{
	if (replay)
	{
		if (cur_frame >= recording_data.size())
		{
			cur_frame = 0;
			replay = false;
			recording_data.clear();
			return InputRecord();
		}
		return recording_data[cur_frame++];
	}
	return InputRecord();
}



sf::Music *current_music = nullptr;

void Scene::SetCurrentMusic(sf::Music *new_music)
{
	if (current_music != new_music)
	{
		StopMusic();
		current_music = new_music;
		current_music->play();
		current_music->setVolume(SETTINGS.stg.music_volume);
	}
}

void Scene::StopMusic()
{
	if (current_music != nullptr)
		current_music->stop();
}

static void ModPi(float& a, float b) {
  if (a - b > PI) {
    a -= 2 * PI;
  } else if (a - b < -PI) {
    a += 2 * PI;
  }
}

Scene::Scene(sf::Music* level_music) :
	intro_needs_snap(true),
	play_single(false),
	is_fullrun(false),
	exposure(1.0f),
	cam_mat(Eigen::Matrix4f::Identity()),
	cam_look_x(0.0f),
	cam_look_y(0.0f),
	cam_dist(default_zoom),
	cam_pos(0.0f, 0.0f, 0.0f),
	cam_mode(CamMode::INTRO),
	marble_rad(1.0f),
	marble_pos(0.0f, 0.0f, 0.0f),
	marble_vel(0.0f, 0.0f, 0.0f),
	marble_mat(Eigen::Matrix3f::Identity()),
	flag_pos(0.0f, 0.0f, 0.0f),
	timer(0),
	sum_time(0),
	music(level_music),
	cur_level(0),
	PBR_Enabled(1),
	Refl_Refr_Enabled(1),
	Shadows_Enabled(1),
	Fractal_Iterations(16),
	MarbleType(0),
	free_camera_speed(1.f),
    LIGHT_DIRECTION(Eigen::Vector3f(-0.36, 0.8, 0.48)),
	PBR_METALLIC(0.5),
	PBR_ROUGHNESS(0.4),
	camera_size(0.075),
	cur_ed_mode(DEFAULT),
	level_editor(false),
	gravity(0.005f),
	time(0.f)
{
  ResetCheats();
  frac_params.setOnes();
  frac_params_smooth.setOnes();
  SnapCamera();
  buff_goal.loadFromFile(goal_wav);
  sound_goal.setBuffer(buff_goal);
  buff_bounce1.loadFromFile(bounce1_wav);
  sound_bounce1.setBuffer(buff_bounce1);
  buff_bounce2.loadFromFile(bounce2_wav);
  sound_bounce2.setBuffer(buff_bounce2);
  buff_bounce3.loadFromFile(bounce3_wav);
  sound_bounce3.setBuffer(buff_bounce3);
  buff_shatter.loadFromFile(shatter_wav);
  sound_shatter.setBuffer(buff_shatter);
}
Eigen::Vector3f Scene::GetVelocity() {
	return marble_vel;
}

void Scene::LoadLevel(int level) {
  SetLevel(level);
  marble_pos = level_copy.start_pos;
  marble_rad = level_copy.marble_rad;
  flag_pos = level_copy.end_pos;
  cam_look_x = level_copy.start_look_x;
}

void Scene::SetMarble(float x, float y, float z, float r) {
  marble_rad = r;
  marble_pos = Eigen::Vector3f(x, y, z);
  marble_vel.setZero();
}
void Scene::SetMarbleScale(float r) {
  marble_rad = r;
}

void Scene::SetFlag(float x, float y, float z) {
  flag_pos = Eigen::Vector3f(x, y, z);
}

void Scene::SetLevel(int level) {
  cur_level = level;    
  level_copy = levels.GetLevel(level);
  SetCurrentMusic(levels.GetLevelMusic(level));
}

void Scene::SetMode(CamMode mode) {
  //Don't reset the timer if transitioning to screen saver
  if ((cam_mode == INTRO && mode == SCREEN_SAVER) ||
      (cam_mode == SCREEN_SAVER && mode == INTRO)) {
  } else {
    timer = 0;
    intro_needs_snap = true;
  }
  cam_mode = mode;
}

void Scene::SetResolution(int x, int y)
{
	ResX = x;
	ResY = y;
}

void Scene::SetWindowResolution(int x, int y)
{
	WinX = x;
	WinY = y;
}

int Scene::GetCountdownTime() const {
  if (cam_mode == DEORBIT && timer >= frame_deorbit) {
    return timer - frame_deorbit;
  } else if (cam_mode == MARBLE) {
    return timer + 3*60;
  } else if (cam_mode == GOAL) {
    return final_time + 3*60;
  } else {
    return -1;
  }
}

sf::Vector3f Scene::GetGoalDirection() const {
  Eigen::Vector3f goal_delta = marble_mat.transpose() * (flag_pos - marble_pos);
  goal_delta.y() = 0.0f;
  const float goal_dir = std::atan2(-goal_delta.z(), goal_delta.x());
  const float a = cam_look_x - goal_dir;
  const float b = std::abs(cam_look_y * 2.0f / PI);
  const float d = goal_delta.norm() / marble_rad;
  return sf::Vector3f(a, b, d);
}

sf::Music& Scene::GetCurMusic() const {
  for (int i = 0; i < num_level_music; ++i) {
    if (cur_level < mus_switches[i]) {
      return music[i];
    }
  }
  return music[0];
}

void Scene::StopAllMusic() {
  for (int i = 0; i < num_level_music; ++i) {
    music[i].stop();
  }
  levels.StopAllMusic();
}

bool Scene::IsHighScore(){
  if (cam_mode != GOAL) {
    return false;
  } else {
	  float best = levels.GetBest(cur_level);
    return best == final_time / 60.f;
  }
}

void Scene::StartNewGame() {
  sum_time = 0;
  play_single = false;
  ResetCheats();
  is_fullrun = true;
  SetLevel(0);
  HideObjects();
  StartRecording();
  SetMode(ORBIT);
}

void Scene::StartNextLevel() {
  if (play_single) {
    cam_mode = MARBLE;
	if (PlayNext && levels.LevelExists(level_copy.link_level))
	{
		SetLevel(level_copy.link_level);
		HideObjects();
		SetMode(ORBIT);
		StartRecording();
	}
	else
	{
		ResetLevel();
	}
    
  } else if (cur_level + 1 == num_levels_midpoint && cam_mode != MIDPOINT) {
    cam_mode = MIDPOINT;
  } else if (cur_level + 1 >= num_levels) {
    cam_mode = FINAL;
  } else {
    SetLevel(cur_level + 1);
    HideObjects();
    SetMode(ORBIT);
	StartRecording();
  }
}

void Scene::ReplayLevel(int level)
{
	StopRecording2File("", 0);
	play_single = true;
	is_fullrun = false;
	ResetCheats();
	SetLevel(level);
	StartReplayFromFile(std::string(recordings_folder) + "/" + ConvertSpaces2_(level_copy.txt) + ".bin");
	HideObjects();
	SetMode(ORBIT);
}

void Scene::StartSingle(int level) {
  play_single = true;
  is_fullrun = false;
  ResetCheats();
  SetLevel(level);
  StartRecording();
  HideObjects();
  SetMode(ORBIT);
}



void Scene::StartLevelEditor(int level)
{
	if (level < 0)
	{
		cur_level = -1;
		level_copy = default_level;
	}
	else
	{
		cur_level = level;
		level_copy = levels.GetLevel(level);
	}
	original_level_name = level_copy.txt;
	play_single = true;
	is_fullrun = false;
	ResetCheats();
	level_editor = true;
	SetMode(ORBIT);
	enable_cheats = false;
	free_camera = true;
}

void Scene::ResetCamera()
{
	SetMode(DEORBIT);
	timer = frame_deorbit;
	free_camera_speed = 1.f;
}

void Scene::StartDefault()
{
	cur_level = -1;
	level_copy = default_level;
}

void Scene::ResetLevel() {
  if (cam_mode == MARBLE || play_single) {
    StopRecording2File("", false);
	StartRecording();
	AddResetPadding();
    SetMode(DEORBIT);
    timer = frame_deorbit;
    frac_params = level_copy.params;
    frac_params_smooth = frac_params;
    marble_pos = level_copy.start_pos;
    marble_vel.setZero();
    marble_rad = level_copy.marble_rad;
    marble_mat.setIdentity();
    flag_pos = level_copy.end_pos;
    cam_look_x = level_copy.start_look_x;
    cam_look_x_smooth = cam_look_x;
    cam_pos = cam_pos_smooth;
    cam_dist = default_zoom;
    cam_dist_smooth = cam_dist;
    cam_look_y = -0.3f;
    cam_look_y_smooth = cam_look_y;
  }
}

void Scene::ResetCheats() {
  enable_cheats = false;
  free_camera = false;
  gravity_type = 0;
  param_mod = -1;
  ignore_goal = false;
  hyper_speed = false;
  disable_motion = false;
  zoom_to_scale = false;
}

void Scene::Synchronize()
{
	for (int i = 0; i < 9; i++)
	{
		 frac_params_smooth[i] = frac_params[i] = level_copy.params[i];
	}
}

void Scene::UpdateCamera(float dx, float dy, float dz, bool speedup) {
	time += 1.f / 60.f;
	//Camera update depends on current mode
  gravity = level_copy.gravity;
  const int iters = speedup ? 5 : 1;
  if (cam_mode == INTRO) {
    UpdateIntro(false);
  } else if (cam_mode == SCREEN_SAVER) {
    UpdateIntro(true);
  } else if (cam_mode == ORBIT) {
    for (int i = 0; i < iters; i++) {
      UpdateOrbit();
      if (cam_mode != ORBIT) {
        break;
      }
    }
  } else if (cam_mode == DEORBIT) {
    for (int i = 0; i < iters; i++) {
      UpdateDeOrbit(dx, dy, dz);
      if (cam_mode != DEORBIT) {
        break;
      }
    }
  } else if (cam_mode == MARBLE) {
    UpdateNormal(dx, dy, dz);
  } else if (cam_mode == GOAL || cam_mode == FINAL || cam_mode == MIDPOINT) {
    for (int i = 0; i < iters; i++) {
      UpdateGoal();
      if (cam_mode != GOAL) {
        break;
      }
    }
  }
}

void Scene::UpdateMarble(float dx, float dy) {
  //Ignore other modes
  if (cam_mode != MARBLE) {
    return;
  }

  //Normalize force if too big
  const float mag2 = dx*dx + dy*dy;
  if (mag2 > 1.0f) {
    const float mag = std::sqrt(mag2);
    dx /= mag;
    dy /= mag;
  }

  if (free_camera) {
    cam_pos += cam_mat.block<3,1>(0,2) * (-marble_rad * dy * 0.5f * free_camera_speed);
    cam_pos += cam_mat.block<3, 1>(0,0) * (marble_rad * dx * 0.5f * free_camera_speed);
    cam_pos_smooth = cam_pos_smooth*0.9f + cam_pos*0.1f;
  } else {
    //Apply all physics (gravity and collision)
    bool onGround = false;
    float max_delta_v = 0.0f;
    for (int i = 0; i < num_phys_steps; ++i) {
      float force = marble_rad * gravity / num_phys_steps;
      if (gravity_type == 1) { force *= 0.25f; } else if (gravity_type == 2) { force *= 4.0f; }
      if (level_copy.planet) {
        marble_vel -= marble_pos.normalized() * force;
      } else {
        marble_vel.y() -= force;
      }
      marble_pos += marble_vel / num_phys_steps;
      onGround |= MarbleCollision(max_delta_v);
    }

    //Play bounce sound if needed
    float bounce_delta_v = max_delta_v / marble_rad;
    if (bounce_delta_v > 0.5f) {
      sound_bounce1.play();
    } else if (bounce_delta_v > 0.25f) {
      sound_bounce2.play();
    } else if (bounce_delta_v > 0.1f) {
      sound_bounce3.setVolume(SETTINGS.stg.fx_volume * (bounce_delta_v / 0.25f));
      sound_bounce3.play();
    }

    //Add force from keyboard
    float f = marble_rad * (onGround ? ground_force : air_force);
    if (hyper_speed) { f *= 4.0f; }
    const float cs = std::cos(cam_look_x);
    const float sn = std::sin(cam_look_x);
    const Eigen::Vector3f v(dx*cs - dy*sn, 0.0f, -dy*cs - dx*sn);
    marble_vel += (marble_mat * v) * f;

    //Apply friction
    marble_vel *= (onGround ? ground_friction : air_friction);
  }

  //Update animated fractals
  if (!disable_motion) {
    frac_params[1] = level_copy.params[1] + level_copy.anim_1 * std::sin(timer * 0.015f);
    frac_params[2] = level_copy.params[2] + level_copy.anim_2 * std::sin(timer * 0.015f);
    frac_params[4] = level_copy.params[4] + level_copy.anim_3 * std::sin(timer * 0.015f);
  }
  frac_params_smooth = frac_params;

  //Check if marble has hit flag post
  if (cam_mode != GOAL && !ignore_goal) {
    const bool flag_y_match = level_copy.planet ?
      marble_pos.y() <= flag_pos.y() && marble_pos.y() >= flag_pos.y() - 7*marble_rad :
      marble_pos.y() >= flag_pos.y() && marble_pos.y() <= flag_pos.y() + 7*marble_rad;
    if (flag_y_match) {
      const float fx = marble_pos.x() - flag_pos.x();
      const float fz = marble_pos.z() - flag_pos.z();
      if (fx*fx + fz*fz < 6 * marble_rad*marble_rad) {
        final_time = timer;
		bool best = false;
        if (!enable_cheats) {
			best = levels.UpdateScore(cur_level, final_time/60.f);
        }
		StopRecording2File(std::string(recordings_folder) + "/" + ConvertSpaces2_(level_copy.txt) + ".bin", best);
        SetMode(GOAL);
        sound_goal.play();
      }
    }
  }

  //Check if marble passed the death barrier
  if (marble_pos.y() < (enable_cheats ? -999.0f : level_copy.kill_y)) {
    ResetLevel();
  }
}

void Scene::UpdateIntro(bool ssaver) {
  //Update the timer
  const float t = -2.0f + timer * 0.002f;
  timer += 1;

  //Get rotational parameters
  const float dist = (ssaver ? 10.0f : 8.0f);
  const Eigen::Vector3f orbit_pt(0.0f, 3.0f, 0.0f);
  const Eigen::Vector3f perp_vec(std::sin(t), 0.0f, std::cos(t));
  cam_pos = orbit_pt + perp_vec * dist;
  cam_pos_smooth = cam_pos_smooth*0.9f + cam_pos*0.1f;

  //Solve for the look direction
  cam_look_x = std::atan2(perp_vec.x(), perp_vec.z());
  if (!ssaver) { cam_look_x += 0.5f; }
  ModPi(cam_look_x_smooth, cam_look_x);
  cam_look_x_smooth = cam_look_x_smooth*0.9f + cam_look_x*0.1f;

  //Update look y
  cam_look_y = (ssaver ? -0.25f : -0.41f);
  cam_look_y_smooth = cam_look_y_smooth*0.9f + cam_look_y*0.1f;

  //Update the camera matrix
  marble_mat.setIdentity();
  MakeCameraRotation();
  cam_mat.block<3, 1>(0, 3) = cam_pos_smooth;

  //Update demo fractal
  frac_params[0] = 1.6f;
  frac_params[1] = 2.0f + 0.5f*std::cos(timer * 0.0021f);
  frac_params[2] = PI + 0.5f*std::cos(timer * 0.000287f);
  frac_params[3] = -4.0f + 0.5f*std::sin(timer * 0.00161f);
  frac_params[4] = -1.0f + 0.1f*std::sin(timer * 0.00123f);
  frac_params[5] = -1.0f + 0.1f*std::cos(timer * 0.00137f);
  frac_params[6] = -0.2f;
  frac_params[7] = -0.1f;
  frac_params[8] = -0.6f;
  frac_params_smooth = frac_params;

  //Make sure marble and flag are hidden
  HideObjects();

  if (intro_needs_snap) {
    SnapCamera();
    intro_needs_snap = false;
  }
}

void Scene::UpdateOrbit() {
  //Update the timer
  const float t = timer * orbit_speed;
  float a = std::min(float(timer) / float(frame_transition), 1.0f);
  a *= a/(2*a*(a - 1) + 1);
  timer += 1;
  sum_time += 1;

  //Get marble location and rotational parameters
  const float orbit_dist = level_copy.orbit_dist;
  const Eigen::Vector3f orbit_pt(0.0f, orbit_dist, 0.0f);
  const Eigen::Vector3f perp_vec(std::sin(t), 0.0f, std::cos(t));
  cam_pos = orbit_pt + perp_vec * (orbit_dist * 2.5f);
  cam_pos_smooth = cam_pos_smooth*orbit_smooth + cam_pos*(1 - orbit_smooth);

  //Solve for the look direction
  cam_look_x = std::atan2(cam_pos_smooth.x(), cam_pos_smooth.z());
  ModPi(cam_look_x_smooth, cam_look_x);
  cam_look_x_smooth = cam_look_x_smooth*(1 - a) + cam_look_x*a;

  //Update look smoothing
  cam_look_y = -0.3f;
  cam_look_y_smooth = cam_look_y_smooth*orbit_smooth + cam_look_y*(1 - orbit_smooth);

  //Update the camera matrix
  marble_mat.setIdentity();
  MakeCameraRotation();
  cam_mat.block<3, 1>(0, 3) = cam_pos_smooth;

  //Update fractal parameters
  ModPi(frac_params[1], level_copy.params[1]);
  ModPi(frac_params[2], level_copy.params[2]);
  frac_params_smooth = frac_params * (1.0f - a) + level_copy.params * a;

  //When done transitioning display the marble and flag
  if (timer >= frame_transition) {
    marble_pos = level_copy.start_pos;
    marble_rad = level_copy.marble_rad;
    flag_pos = level_copy.end_pos;
  }

  //When done transitioning, setup level
  if (timer >= frame_orbit) {
    frac_params = level_copy.params;
    cam_look_x = cam_look_x_smooth;
    cam_pos = cam_pos_smooth;
    cam_dist = default_zoom;
    cam_dist_smooth = cam_dist;
    cam_mode = DEORBIT;
  }
}

void Scene::UpdateDeOrbit(float dx, float dy, float dz) {
  //Update the timer
  const float t = timer * orbit_speed;
  float b = std::min(float(std::max(timer - frame_orbit, 0)) / float(frame_deorbit - frame_orbit), 1.0f);
  b *= b/(2*b*(b - 1) + 1);
  timer += 1;
  sum_time += 1;

  if (timer > frame_deorbit + 1) {
    UpdateCameraOnly(dx, dy, dz);
  } else {
    //Get marble location and rotational parameters
    const float orbit_dist = level_copy.orbit_dist;
    const Eigen::Vector3f orbit_pt(0.0f, orbit_dist, 0.0f);
    const Eigen::Vector3f perp_vec(std::sin(t), 0.0f, std::cos(t));
    const Eigen::Vector3f orbit_cam_pos = orbit_pt + perp_vec * (orbit_dist * 2.5f);
    cam_pos = cam_pos*orbit_smooth + orbit_cam_pos*(1 - orbit_smooth);

    //Solve for the look direction
    const float start_look_x = level_copy.start_look_x;
    cam_look_x = std::atan2(cam_pos.x(), cam_pos.z());
    ModPi(cam_look_x, start_look_x);

    //Solve for the look direction
    cam_look_x_smooth = cam_look_x*(1 - b) + start_look_x*b;

    //Update look smoothing
    cam_look_y = -0.3f;
    cam_look_y_smooth = cam_look_y_smooth*orbit_smooth + cam_look_y*(1 - orbit_smooth);

    //Update the camera rotation matrix
    MakeCameraRotation();

    //Update the camera position
    Eigen::Vector3f marble_cam_pos = marble_pos + cam_mat.block<3, 3>(0, 0) * Eigen::Vector3f(0.0f, 0.0f, marble_rad * cam_dist_smooth);
    marble_cam_pos += Eigen::Vector3f(0.0f, marble_rad * cam_dist_smooth * 0.1f, 0.0f);
    cam_pos_smooth = cam_pos*(1 - b) + marble_cam_pos*b;
    cam_mat.block<3, 1>(0, 3) = cam_pos_smooth;

    //Required for a smooth transition later on
    cam_look_x = cam_look_x_smooth;
    cam_look_y = cam_look_y_smooth;
  }

  //When done deorbiting, transition to play
  if (timer > frame_countdown) {
    cam_mode = MARBLE;
    cam_pos = cam_pos_smooth;
    timer = 0;
  }
}

void Scene::UpdateCameraOnly(float dx, float dy, float dz) {
  //Update camera zoom
  if (param_mod >= 0) {
    const float new_param = level_copy.params[param_mod] + dz*0.01f;
    level_copy.params[param_mod] = frac_params_smooth[param_mod] = frac_params[param_mod] = new_param;
  } else if (zoom_to_scale) {
    level_copy.marble_rad *= std::pow(2.0f, -dz);
    level_copy.marble_rad = std::min(std::max(level_copy.marble_rad, 0.0006f), 0.6f);
    marble_rad = marble_rad*zoom_smooth + level_copy.marble_rad*(1 - zoom_smooth);
  } else {
    cam_dist *= std::pow(2.0f, -dz);
    cam_dist = std::min(std::max(cam_dist, 5.0f), 30.0f);
  }
  cam_dist_smooth = cam_dist_smooth*zoom_smooth + cam_dist*(1 - zoom_smooth);

  //Update look direction
  cam_look_x += dx;
  cam_look_y += dy;
  cam_look_y = std::min(std::max(cam_look_y, -PI / 2), PI / 2);
  while (cam_look_x > PI) { cam_look_x -= 2 * PI; }
  while (cam_look_x < -PI) { cam_look_x += 2 * PI; }

  //Update look smoothing
  const float a = (free_camera ? look_smooth_free_camera : look_smooth);
  ModPi(cam_look_x_smooth, cam_look_x);
  cam_look_x_smooth = cam_look_x_smooth*a + cam_look_x*(1 - a);
  cam_look_y_smooth = cam_look_y_smooth*a + cam_look_y*(1 - a);

  //Setup rotation matrix for planets
  if (level_copy.planet) {
    marble_mat.col(1) = marble_pos.normalized();
    marble_mat.col(2) = -marble_mat.col(1).cross(marble_mat.col(0)).normalized();
    marble_mat.col(0) = -marble_mat.col(2).cross(marble_mat.col(1)).normalized();
  } else {
    marble_mat.setIdentity();
  }

  //Update the camera matrix
  MakeCameraRotation();
  if (!free_camera) {
    cam_pos = marble_pos + cam_mat.block<3, 3>(0, 0) * Eigen::Vector3f(0.0f, 0.0f, marble_rad * cam_dist_smooth);
    cam_pos += marble_mat.col(1) * (marble_rad * cam_dist_smooth * 0.1f);
    cam_pos_smooth = cam_pos;
  }
  cam_mat.block<3, 1>(0, 3) = cam_pos_smooth;
}

void Scene::UpdateNormal(float dx, float dy, float dz) {
  //Update camera
  UpdateCameraOnly(dx, dy, dz);

  //Update timer
  timer += 1;
  sum_time += 1;
}

void Scene::UpdateGoal() {
  //Update the timer
  const float t = timer * 0.01f;
  float a = std::min(t / 75.0f, 1.0f);
  timer += 1;
  if (cur_level != num_levels_midpoint - 1 && cur_level != num_levels - 1) {
    sum_time += 1;
  }

  //Get marble location and rotational parameters
  const float flag_dist = marble_rad * 6.5f;
  const Eigen::Vector3f orbit_pt = flag_pos + marble_mat * Eigen::Vector3f(0.0f, flag_dist, 0.0f);
  const Eigen::Vector3f perp_vec = Eigen::Vector3f(std::sin(t), 0.0f, std::cos(t));
  cam_pos = orbit_pt + marble_mat * perp_vec * (flag_dist * 3.5f);
  cam_pos_smooth = cam_pos_smooth*(1 - a) + cam_pos*a;

  //Solve for the look direction
  cam_look_x = std::atan2(perp_vec.x(), perp_vec.z());
  ModPi(cam_look_x_smooth, cam_look_x);
  cam_look_x_smooth = cam_look_x_smooth*(1 - a) + cam_look_x*a;

  //Update look smoothing
  cam_look_y = -0.25f;
  cam_look_y_smooth = cam_look_y_smooth*0.99f + cam_look_y*(1 - 0.99f);

  //Update the camera matrix
  MakeCameraRotation();
  cam_mat.block<3, 1>(0, 3) = cam_pos_smooth;

  //Animate marble
  marble_vel += (orbit_pt - marble_pos) * 0.005f;
  marble_pos += marble_vel;
  if (marble_vel.norm() > marble_rad*0.02f) {
    marble_vel *= 0.95f;
  }

  if (timer > 300 && cam_mode != FINAL && cam_mode != MIDPOINT) {
    StartNextLevel();
  }
}

void Scene::MakeCameraRotation() {
  cam_mat.setIdentity();
  const Eigen::AngleAxisf aa_x_smooth(cam_look_x_smooth, Eigen::Vector3f::UnitY());
  const Eigen::AngleAxisf aa_y_smooth(cam_look_y_smooth, Eigen::Vector3f::UnitX());
  cam_mat.block<3, 3>(0, 0) = marble_mat * (aa_x_smooth * aa_y_smooth).toRotationMatrix();
}

void Scene::SnapCamera() {
  cam_look_x_smooth = cam_look_x;
  cam_look_y_smooth = cam_look_y;
  cam_dist_smooth = cam_dist;
  cam_pos_smooth = cam_pos;
}

void Scene::HideObjects() {
  marble_pos = Eigen::Vector3f(999.0f, 999.0f, 999.0f);
  flag_pos = Eigen::Vector3f(999.0f, 999.0f, 999.0f);
  marble_vel.setZero();
}

void Scene::Write(sf::Shader& shader) const {
  shader.setUniform("iMat", sf::Glsl::Mat4(cam_mat.data()));

  if (level_editor)
  {
	  shader.setUniform("iMarblePos", sf::Glsl::Vec3(level_copy.start_pos.x(), level_copy.start_pos.y(), level_copy.start_pos.z()));
	  shader.setUniform("iFlagPos",  sf::Glsl::Vec3(level_copy.end_pos.x(), level_copy.end_pos.y(), level_copy.end_pos.z()));
  }
  else
  {
	  shader.setUniform("iMarblePos", free_camera ?
		  sf::Glsl::Vec3(999.0f, 999.0f, 999.0f) :
		  sf::Glsl::Vec3(marble_pos.x(), marble_pos.y(), marble_pos.z())
	  );
	  shader.setUniform("iFlagPos", free_camera ?
		  sf::Glsl::Vec3(-999.0f, -999.0f, -999.0f) :
		  sf::Glsl::Vec3(flag_pos.x(), flag_pos.y(), flag_pos.z())
	  );
  }

  if (cam_mode != INTRO)
  {
	  shader.setUniform("LIGHT_DIRECTION", sf::Glsl::Vec3(level_copy.light_dir[0], level_copy.light_dir[1], level_copy.light_dir[2]));
	  shader.setUniform("PBR_ENABLED", PBR_Enabled);
	  shader.setUniform("PBR_METALLIC", level_copy.PBR_metal);
	  shader.setUniform("PBR_ROUGHNESS", level_copy.PBR_roughness);
  }
  else
  {
	  shader.setUniform("LIGHT_DIRECTION", sf::Glsl::Vec3(LIGHT_DIRECTION[0], LIGHT_DIRECTION[1], LIGHT_DIRECTION[2]));
	  shader.setUniform("PBR_ENABLED", PBR_Enabled);
	  shader.setUniform("PBR_METALLIC", PBR_METALLIC);
	  shader.setUniform("PBR_ROUGHNESS", PBR_ROUGHNESS);
  }

  shader.setUniform("BACKGROUND_COLOR", sf::Glsl::Vec3(level_copy.background_col[0], level_copy.background_col[1], level_copy.background_col[2]));
  shader.setUniform("LIGHT_COLOR", sf::Glsl::Vec3(level_copy.light_col[0], level_copy.light_col[1], level_copy.light_col[2]));

  shader.setUniform("iMarbleRad", level_copy.marble_rad);

  shader.setUniform("iFlagScale", level_copy.planet ? -level_copy.marble_rad : level_copy.marble_rad);

  shader.setUniform("iFracScale", frac_params_smooth[0]);
  shader.setUniform("iFracAng1", frac_params_smooth[1]);
  shader.setUniform("iFracAng2", frac_params_smooth[2]);
  shader.setUniform("iFracShift", sf::Glsl::Vec3(frac_params_smooth[3], frac_params_smooth[4], frac_params_smooth[5]));
  shader.setUniform("iFracCol", sf::Glsl::Vec3(frac_params_smooth[6], frac_params_smooth[7], frac_params_smooth[8]));

  shader.setUniform("iExposure", exposure);


  shader.setUniform("SHADOWS_ENABLED", Shadows_Enabled);
  shader.setUniform("CAMERA_SIZE", camera_size*level_copy.marble_rad/0.035f);
  shader.setUniform("FRACTAL_ITER", level_copy.FractalIter);
  shader.setUniform("REFL_REFR_ENABLED", Refl_Refr_Enabled);
  shader.setUniform("MARBLE_MODE", MarbleType);
}



void Scene::WriteRenderer(Renderer & rd)
{
	//Update the camera
	vec3 cam_pos = vec3(cam_mat(0, 3), cam_mat(1, 3), cam_mat(2, 3));
	vec3 dirx = vec3(cam_mat(0, 0), cam_mat(1, 0), cam_mat(2, 0));
	vec3 diry = vec3(cam_mat(0, 1), cam_mat(1, 1), cam_mat(2, 1));
	vec3 dirz = -vec3(cam_mat(0, 2), cam_mat(1, 2), cam_mat(2, 2));
	rd.camera.SetPosition(cam_pos);
	rd.camera.SetDirX(dirx);
	rd.camera.SetDirY(diry);
	rd.camera.SetDirZ(dirz);
	rd.camera.SetCameraSize(camera_size*level_copy.marble_rad / 0.035f);
	rd.camera.eye_separation = SETTINGS.stg.eye_separation*level_copy.marble_rad / 0.035f;
	//write all the uniform values to the rendering pipeline
	for (auto &shader : rd.shader_pipeline)
	{
		WriteShader(shader);
	}
}

void Scene::WriteShader(ComputeShader& shader)
{

	if (level_editor)
	{
		shader.setUniform("iMarblePos", vec3(level_copy.start_pos.x(), level_copy.start_pos.y(), level_copy.start_pos.z()));
		shader.setUniform("iFlagPos", vec3(level_copy.end_pos.x(), level_copy.end_pos.y(), level_copy.end_pos.z()));
	}
	else
	{
		shader.setUniform("iMarblePos", free_camera ?
			vec3(999.0f, 999.0f, 999.0f) :
			vec3(marble_pos.x(), marble_pos.y(), marble_pos.z())
		);
		shader.setUniform("iFlagPos", free_camera ?
			vec3(-999.0f, -999.0f, -999.0f) :
			vec3(flag_pos.x(), flag_pos.y(), flag_pos.z())
		);
	}

	if (cam_mode != INTRO)
	{
		shader.setUniform("LIGHT_DIRECTION", vec3(level_copy.light_dir[0], level_copy.light_dir[1], level_copy.light_dir[2]));
		shader.setUniform("PBR_ENABLED", PBR_Enabled);
		shader.setUniform("PBR_METALLIC", level_copy.PBR_metal);
		shader.setUniform("PBR_ROUGHNESS", level_copy.PBR_roughness);
	}
	else
	{
		shader.setUniform("LIGHT_DIRECTION", vec3(LIGHT_DIRECTION[0], LIGHT_DIRECTION[1], LIGHT_DIRECTION[2]));
		shader.setUniform("PBR_ENABLED", PBR_Enabled);
		shader.setUniform("PBR_METALLIC", PBR_METALLIC);
		shader.setUniform("PBR_ROUGHNESS", PBR_ROUGHNESS);
	}

	shader.setUniform("BACKGROUND_COLOR", vec3(level_copy.background_col[0], level_copy.background_col[1], level_copy.background_col[2]));
	shader.setUniform("LIGHT_COLOR", vec3(level_copy.light_col[0], level_copy.light_col[1], level_copy.light_col[2]));

	shader.setUniform("iMarbleRad", level_copy.marble_rad);
	shader.setUniform("iFlagScale", level_copy.planet ? -level_copy.marble_rad : level_copy.marble_rad);

	shader.setUniform("iFracScale", frac_params_smooth[0]);
	shader.setUniform("iFracAng1", frac_params_smooth[1]);
	shader.setUniform("iFracAng2", frac_params_smooth[2]);
	shader.setUniform("iFracShift", vec3(frac_params_smooth[3], frac_params_smooth[4], frac_params_smooth[5]));
	shader.setUniform("iFracCol", vec3(frac_params_smooth[6], frac_params_smooth[7], frac_params_smooth[8]));

	shader.setUniform("SHADOWS_ENABLED", Shadows_Enabled);
	shader.setUniform("FOG_ENABLED", Fog_Enabled);
	shader.setUniform("FRACTAL_ITER", level_copy.FractalIter);
	shader.setUniform("REFL_REFR_ENABLED", Refl_Refr_Enabled);
	shader.setUniform("MARBLE_MODE", MarbleType);
	shader.setUniform("FRACTAL_GLOW", SETTINGS.stg.fractal_glow);
	shader.setUniform("FLAG_GLOW", SETTINGS.stg.flag_glow);

	shader.setUniform("gamma_material", gamma_material);
	shader.setUniform("gamma_sky", gamma_sky);
	shader.setUniform("gamma_camera", gamma_camera);

	shader.setUniform("time", time);
}

std::unique_ptr<Fractal> Scene::Frac() const {
  const float frac_scale = frac_params_smooth[0];
  const float frac_angle1 = frac_params_smooth[1];
  const float frac_angle2 = frac_params_smooth[2];
  const Eigen::Vector3f frac_shift = frac_params_smooth.segment<3>(3);

  std::vector<std::unique_ptr<FoldableBase>> inner_folds{};
  inner_folds.emplace_back(std::make_unique<FoldAbs>());
  inner_folds.emplace_back(std::make_unique<FoldRotate>(Axis::Z, frac_angle1));
  inner_folds.emplace_back(std::make_unique<FoldMenger>());
  inner_folds.emplace_back(std::make_unique<FoldRotate>(Axis::X, frac_angle2));
  inner_folds.emplace_back(std::make_unique<FoldScaleTranslate>(frac_scale, frac_shift));

  auto series = std::make_unique<FoldSeries>(std::move(inner_folds));
  auto loop = std::make_unique<FoldRepeat>(level_copy.FractalIter, std::move(series));

  return std::make_unique<Fractal>(std::move(loop),
      std::make_unique<ObjectBox>(Eigen::Vector3f(6.0, 6.0, 6.0))
  );
}

//Hard-coded to match the fractal
float Scene::DE(const Eigen::Vector3f& pt) const {
  //Easier to work with names
  const float frac_scale = frac_params_smooth[0];
  const float frac_angle1 = frac_params_smooth[1];
  const float frac_angle2 = frac_params_smooth[2];
  const Eigen::Vector3f frac_shift = frac_params_smooth.segment<3>(3);
  const Eigen::Vector3f frac_color = frac_params_smooth.segment<3>(6);

  Eigen::Vector4f p;
  p << pt, 1.0f;
  float final_value_copy = Frac()->DistanceEstimator(p);

  for (int i = 0; i < level_copy.FractalIter; ++i) {
    //absFold
    p.segment<3>(0) = p.segment<3>(0).cwiseAbs();
    //rotZ
    const float rotz_c = std::cos(frac_angle1);
    const float rotz_s = std::sin(frac_angle1);
    const float rotz_x = rotz_c*p.x() + rotz_s*p.y();
    const float rotz_y = rotz_c*p.y() - rotz_s*p.x();
    p.x() = rotz_x; p.y() = rotz_y;
    //mengerFold
    float a = std::min(p.x() - p.y(), 0.0f);
    p.x() -= a; p.y() += a;
    a = std::min(p.x() - p.z(), 0.0f);
    p.x() -= a; p.z() += a;
    a = std::min(p.y() - p.z(), 0.0f);
    p.y() -= a; p.z() += a;
    //rotX
    const float rotx_c = std::cos(frac_angle2);
    const float rotx_s = std::sin(frac_angle2);
    const float rotx_y = rotx_c*p.y() + rotx_s*p.z();
    const float rotx_z = rotx_c*p.z() - rotx_s*p.y();
    p.y() = rotx_y; p.z() = rotx_z;
    //scaleTrans
    p *= frac_scale;
    p.segment<3>(0) += frac_shift;
  }
  const Eigen::Vector3f a = p.segment<3>(0).cwiseAbs() - Eigen::Vector3f(6.0f, 6.0f, 6.0f);
  float final_val = (std::min(std::max(std::max(a.x(), a.y()), a.z()), 0.0f) + a.cwiseMax(0.0f).norm()) / p.w();
  assert(final_val == final_value_copy);
  return final_val;
}

//Hard-coded to match the fractal
Eigen::Vector3f Scene::NP(const Eigen::Vector3f& pt) const {
  //Easier to work with names
  const float frac_scale = frac_params_smooth[0];
  const float frac_angle1 = frac_params_smooth[1];
  const float frac_angle2 = frac_params_smooth[2];
  const Eigen::Vector3f frac_shift = frac_params_smooth.segment<3>(3);
  const Eigen::Vector3f frac_color = frac_params_smooth.segment<3>(6);

  static std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> p_hist;
  p_hist.clear();
  Eigen::Vector4f p;
  p << pt, 1.0f;
  auto np = Frac()->NearestPoint(p);
  //Fold the point, keeping history
  for (int i = 0; i < level_copy.FractalIter; ++i) {
    //absFold
    p_hist.push_back(p);
    p.segment<3>(0) = p.segment<3>(0).cwiseAbs();
    //rotZ
    const float rotz_c = std::cos(frac_angle1);
    const float rotz_s = std::sin(frac_angle1);
    const float rotz_x = rotz_c*p.x() + rotz_s*p.y();
    const float rotz_y = rotz_c*p.y() - rotz_s*p.x();
    p.x() = rotz_x; p.y() = rotz_y;
    //mengerFold
    p_hist.push_back(p);
    float a = std::min(p.x() - p.y(), 0.0f);
    p.x() -= a; p.y() += a;
    a = std::min(p.x() - p.z(), 0.0f);
    p.x() -= a; p.z() += a;
    a = std::min(p.y() - p.z(), 0.0f);
    p.y() -= a; p.z() += a;
    //rotX
    const float rotx_c = std::cos(frac_angle2);
    const float rotx_s = std::sin(frac_angle2);
    const float rotx_y = rotx_c*p.y() + rotx_s*p.z();
    const float rotx_z = rotx_c*p.z() - rotx_s*p.y();
    p.y() = rotx_y; p.z() = rotx_z;
    //scaleTrans
    p *= frac_scale;
    p.segment<3>(0) += frac_shift;
  }
  //Get the nearest point
  Eigen::Vector3f n = p.segment<3>(0).cwiseMax(-6.0f).cwiseMin(6.0f);
  //Then unfold the nearest point (reverse order)
  for (int i = 0; i < level_copy.FractalIter; ++i) {
    //scaleTrans
    n.segment<3>(0) -= frac_shift;
    n /= frac_scale;
    //rotX
    const float rotx_c = std::cos(-frac_angle2);
    const float rotx_s = std::sin(-frac_angle2);
    const float rotx_y = rotx_c*n.y() + rotx_s*n.z();
    const float rotx_z = rotx_c*n.z() - rotx_s*n.y();
    n.y() = rotx_y; n.z() = rotx_z;
    //mengerUnfold
    p = p_hist.back(); p_hist.pop_back();
    const float mx = std::max(p[0], p[1]);
    if (std::min(p[0], p[1]) < std::min(mx, p[2])) {
      std::swap(n[1], n[2]);
    }
    if (mx < p[2]) {
      std::swap(n[0], n[2]);
    }
    if (p[0] < p[1]) {
      std::swap(n[0], n[1]);
    }
    //rotZ
    const float rotz_c = std::cos(-frac_angle1);
    const float rotz_s = std::sin(-frac_angle1);
    const float rotz_x = rotz_c*n.x() + rotz_s*n.y();
    const float rotz_y = rotz_c*n.y() - rotz_s*n.x();
    n.x() = rotz_x; n.y() = rotz_y;
    //absUnfold
    p = p_hist.back(); p_hist.pop_back();
    if (p[0] < 0.0f) {
      n[0] = -n[0];
    }
    if (p[1] < 0.0f) {
      n[1] = -n[1];
    }
    if (p[2] < 0.0f) {
      n[2] = -n[2];
    }
  }
  assert(np == n);
  return n;
}

bool Scene::MarbleCollision(float& delta_v) {
  //Check if the distance estimate indicates a collision
  const float de = DE(marble_pos);
  if (de >= marble_rad) {
    return de < marble_rad * ground_ratio;
  }

  //Check if the marble has been crushed by the fractal
  if (de < marble_rad * 0.001f) {
    sound_shatter.play();
    marble_pos.y() = -9999.0f;
    return false;
  }

  //Find the nearest point and compute offset
  const Eigen::Vector3f np = NP(marble_pos);
  const Eigen::Vector3f d = np - marble_pos;
  const Eigen::Vector3f dn = d.normalized();

  //Apply the offset to the marble's position and velocity
  const float dv = marble_vel.dot(dn);
  delta_v = std::max(delta_v, dv);
  marble_pos -= dn * marble_rad - d;
  marble_vel -= dn * (dv * marble_bounce);
  return true;
}

void Scene::Cheat_ColorChange() {
  if (!enable_cheats) { return; }
  level_copy.params[6] = frac_params_smooth[6] = frac_params[6] = float((rand() % 201) - 100) * 0.01f;
  level_copy.params[7] = frac_params_smooth[7] = frac_params[7] = float((rand() % 201) - 100) * 0.01f;
  level_copy.params[8] = frac_params_smooth[8] = frac_params[8] = float((rand() % 201) - 100) * 0.01f;
}
void Scene::Cheat_FreeCamera() {
  if (!enable_cheats) { return; }
  free_camera = !free_camera;
}
void Scene::Cheat_Gravity() {
  if (!enable_cheats) { return; }
  gravity_type = (gravity_type + 1) % 3;
}
void Scene::Cheat_HyperSpeed() {
  if (!enable_cheats) { return; }
  hyper_speed = !hyper_speed;
}
void Scene::Cheat_IgnoreGoal() {
  if (!enable_cheats) { return; }
  ignore_goal = !ignore_goal;
}
void Scene::Cheat_Motion() {
  if (!enable_cheats) { return; }
  disable_motion = !disable_motion;
}
void Scene::Cheat_Planet() {
  if (!enable_cheats) { return; }
  level_copy.planet = !level_copy.planet;
}
void Scene::Cheat_Zoom() {
  if (!enable_cheats) { return; }
  zoom_to_scale = !zoom_to_scale;
}
void Scene::Cheat_Param(int param) {
  if (!enable_cheats) { return; }
  param_mod = param;
}

void Scene::ExitEditor()
{
	levels.StopAllMusic();
	level_editor = false;
}


Eigen::Vector3f Scene::MouseRayCast(int mousex, int mousey, float min_dist)
{
	Eigen::Vector2f screen_pos = Eigen::Vector2f((float)mousex / (float)WinX,1.f - (float)mousey/ (float)WinY);

	std::cerr << screen_pos << std::endl;
	Eigen::Vector2f uv = 2 * screen_pos - Eigen::Vector2f(1.f, 1.f);
	uv.x() *= (float)ResX / (float)ResY;

	//Convert screen coordinate to 3d ray
	Eigen::Vector4f v1 = Eigen::Vector4f(uv.x(), uv.y(), -FOCAL_DIST, 0.0);
	v1.normalize();
	Eigen::Vector4f v2 = Eigen::Vector4f(camera_size*uv.x(), camera_size*uv.y(), 0, 1);
	Eigen::Vector4f ray = cam_mat * v1;
	Eigen::Vector4f p = cam_mat * v2;

	return RayMarch(Eigen::Vector3f(p[0], p[1], p[2]), Eigen::Vector3f(ray[0], ray[1], ray[2]), min_dist);
}

Eigen::Vector3f Scene::RayMarch(const Eigen::Vector3f & pt, const Eigen::Vector3f & ray, float min_dist)
{
	float td = 0;
	for (int i = 0; i < MAX_MARCHES && td < MAX_DIST; i++)
	{
		float de = DE(pt + td * ray);
		if (de < min_dist)
		{
			break;
		}
		td += de;
	}
	return pt + td * ray;
}
