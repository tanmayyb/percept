/***************** Random Obstacle Heuristic *****************/

/***********************************************
 * mass_dist_vec = robot_obstacle_vec: vector from robot to obstacle
 * mass_rvel_vec = rel_vel: relative velocity between robot and obstacle
 * dist_to_mass = dist_obs: distance between robot and obstacle
 * force_vec = curr_force: force vector
 * mass_dist_vec_normalized = cfagent_to_obs: normalized vector from robot to obstacle
 */


void RealCfAgent::circForce(const std::vector<Obstacle> &obstacles,
                            const double k_circ, const CfAgent &agent) 
{
  Eigen::Vector3d goal_vec{g_pos_ - getLatestPosition()};
  for (int i = 0; i < obstacles.size() - 1; i++) 
{
    Eigen::Vector3d robot_obstacle_vec{obstacles.at(i).getPosition()- getLatestPosition()};
    Eigen::Vector3d rel_vel{vel_ - obstacles.at(i).getVelocity()};
    if (robot_obstacle_vec.normalized().dot(goal_vec.normalized()) < -0.01 &&
        robot_obstacle_vec.dot(rel_vel) < -0.01) 
        {
      continue;
    }
    double dist_obs{
        (getLatestPosition() - obstacles.at(i).getPosition()).norm() -
        (rad_ + obstacles.at(i).getRadius())};
    dist_obs = std::max(dist_obs, 1e-5);
    Eigen::Vector3d curr_force{0.0, 0.0, 0.0};
    Eigen::Vector3d current;
    if (dist_obs < detect_shell_rad_) 
    {
      if (!known_obstacles_.at(i)) 
      {
        field_rotation_vecs_.at(i) = calculateRotationVector(getLatestPosition(), getGoalPosition(), obstacles, i, agent);
        known_obstacles_.at(i) = true;
      }
      double vel_norm = rel_vel.norm();
      if (vel_norm != 0) 
      {
        Eigen::Vector3d normalized_vel = rel_vel / vel_norm;
        current = currentVector(getLatestPosition(), rel_vel, getGoalPosition(),obstacles, i, field_rotation_vecs_, agent);
        curr_force = (k_circ / pow(dist_obs, 2)) * normalized_vel.cross(current.cross(normalized_vel));
      }
    }
    force_ += curr_force;
  }
}



Eigen::Vector3d RandomCfAgent::currentVector(
    const Eigen::Vector3d agent_pos, const Eigen::Vector3d agent_vel,
    const Eigen::Vector3d goal_pos, const std::vector<Obstacle> &obstacles,
    const int obstacle_id,
    const std::vector<Eigen::Vector3d> field_rotation_vecs) const {
  Eigen::Vector3d cfagent_to_obs{obstacles[obstacle_id].getPosition() -
                                 agent_pos};
  cfagent_to_obs.normalize();
  Eigen::Vector3d current =
      cfagent_to_obs.cross(field_rotation_vecs.at(obstacle_id));
  current.normalize();
  return current;
}

Eigen::Vector3d RandomCfAgent::calculateRotationVector(
    const Eigen::Vector3d agent_pos, const Eigen::Vector3d goal_pos,
    const std::vector<Obstacle> &obstacles, const int obstacle_id) const {
  Eigen::Vector3d goal_vec{goal_pos - agent_pos};
  goal_vec.normalize();
  Eigen::Vector3d rot_vec = goal_vec.cross(random_vecs_.at(obstacle_id));
  return rot_vec;
}


Eigen::Vector3d RandomCfAgent::makeRandomVector() const 
{
  Eigen::Vector3d ret;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  ret << dis(gen), dis(gen), dis(gen);
  return ret;
}
