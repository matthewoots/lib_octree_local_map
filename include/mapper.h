/*
 * mapper.h
 *
 * ---------------------------------------------------------------------
 * Copyright (C) 2022 Matthew (matthewoots at gmail.com)
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * ---------------------------------------------------------------------
 *
 * 
 * 
 */
/* 
* Some Documentations
* https://pointclouds.org/documentation/classpcl_1_1octree_1_1_octree_point_cloud.html
* https://pcl.readthedocs.io/projects/tutorials/en/latest/octree.html
* https://github.com/otherlab/pcl/blob/master/test/octree/test_octree.cpp 
*/

#ifndef MAPPER_H
#define MAPPER_H

#include <iostream>
#include <math.h>
#include <vector>
#include <chrono>
#include <mutex>
#include <Eigen/Core>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

// #include <pcl/octree/octree.h>
#include <pcl/octree/octree_search.h>

using namespace std::chrono;

#define KNRM  "\033[0m"
#define KRED  "\033[31m"
#define KGRN  "\033[32m"
#define KYEL  "\033[33m"
#define KBLU  "\033[34m"
#define KMAG  "\033[35m"
#define KCYN  "\033[36m"
#define KWHT  "\033[37m"

namespace octree_map
{
    class sliding_map
    {
        public:

            /** 
            * @brief Global map parameters
            * @param use_sensor fuse pcl from sensor or global map
            * @param s_r sensor_range provided
            * @param r resolution of the generated map
            * @param radius radius of the sliding map
            * @param mn_b minimum boundary for the octree
            * @param mx_b maximum boundary for the octree
            * @param m_k max key for the octree
            **/
            struct map_parameters 
            {
                bool use_sensor; // @fuse pcl from sensor or global
                double s_r; // @sensor_range
                double r; // @resolution
                double radius; // @radius
                Eigen::Vector3d mn_b; // minimum boundary for the octree
                Eigen::Vector3d mx_b; // maximum boundary for the octree
                size_t m_k[3]; // max key for the octree
            };

            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree_global = 
                decltype(octree_global)(0.1);
            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree_local = 
                decltype(octree_local)(0.1);

            pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud;

            pcl::PointCloud<pcl::PointXYZ>::VectorType occupied_voxels;

            map_parameters global_param;

            int occupied_size = 0;
            
            bool initialized;

            std::vector<Eigen::Vector3d> sensing_offset;

            std::mutex update_pose_mutex;

            /** 
             * @brief mapper
             * Constructor for mapper
            **/ 
            sliding_map(){};

            /** 
             * @brief ~mapper
             * Destructor for mapper
            **/ 
            ~sliding_map()
            {
                octree_global.deleteTree();
                octree_local.deleteTree();
            }

            /** 
             * @brief set_parameters
             * Setup the parameters for the mapping module
            **/ 
            void set_parameters(
                double hfov, double vfov, double resolution, 
                double sensor_range, double map_size,
                pcl::PointCloud<pcl::PointXYZ>::Ptr obs_pcl, 
                bool use_sensor);

            /** 
             * @brief get_sliding_map_from_global
             * If there is a global map present, use this
            **/
            pcl::PointCloud<pcl::PointXYZ>::Ptr 
                get_sliding_map_from_global(
                Eigen::Vector3d p, Eigen::Quaterniond q);
            
            /** 
             * @brief get_sliding_map_from_sensor
             * If there is only sensor data present, use this
            **/
            pcl::PointCloud<pcl::PointXYZ>::Ptr 
                get_sliding_map_from_sensor(
                pcl::PointCloud<pcl::PointXYZ>::Ptr pcl,    
                Eigen::Vector3d p, Eigen::Quaterniond q);

        private:

            /** 
             * @brief check_approx_intersection_by_segment
             * Checks for intersected voxels that contain pointclouds 
             * Edit function from:
             * https://pointclouds.org/documentation/octree__pointcloud_8hpp_source.html#l00269
             * Definition at line 269 of file octree_pointcloud.hpp
            **/ 
            bool check_approx_intersection_by_segment(
                const Eigen::Vector3d origin, const Eigen::Vector3d end, 
                Eigen::Vector3d& intersect);
            
            /** 
             * @brief gen_leaf_node_center_from_octree_key
             * Edited from the protected function for octree
             * void pcl::octree::OctreePointCloud<PointT, LeafContainerT, BranchContainerT, OctreeT>::
             * genLeafNodeCenterFromOctreeKey(const OctreeKey& key, PointT& point) const
            **/
            void gen_leaf_node_center_from_octree_key(
                const pcl::octree::OctreeKey key, pcl::PointXYZ& point);
            
            /** 
             * @brief gen_octree_key_for_point
             * Edited from the protected function for octree
             * void pcl::octree::OctreePointCloud<PointT, LeafContainerT, BranchContainerT, OctreeT>::
             * genOctreeKeyforPoint(const PointT& point_arg, OctreeKey& key_arg) const
            **/
            void gen_octree_key_for_point(
                const pcl::PointXYZ point_arg, 
                pcl::octree::OctreeKey& key_arg);

            /** 
             * @brief get_raycast_on_pcl
            **/
            pcl::PointCloud<pcl::PointXYZ>::Ptr 
                get_raycast_on_pcl(
                Eigen::Vector3d p, Eigen::Quaterniond q);

            void get_estimated_center_of_point(
                Eigen::Vector3d p, Eigen::Vector3d &est);
            
            void extract_point_cloud_within_boundary(
                Eigen::Vector3d c);

            void get_bresenham_3d_from_origin( 
                const int x2, const int y2, const int z2, 
                std::vector<Eigen::Vector3i> &idx);

            /** 
             * @brief is_point_within_octree
             * Check if the point is within the octree 
            **/
            bool is_point_within_octree(
                Eigen::Vector3d point)
            {
                // Check octree boundary
                if (point.x() < global_param.mx_b.x() - global_param.r/2 && 
                    point.x() > global_param.mn_b.x() + global_param.r/2 &&
                    point.y() < global_param.mx_b.y() - global_param.r/2 && 
                    point.y() > global_param.mn_b.y() + global_param.r/2 &&
                    point.z() < global_param.mx_b.z() - global_param.r/2 && 
                    point.z() > global_param.mn_b.z() + global_param.r/2)
                    return true;
                else
                    return false;
            }

            /** 
             * @brief get_line_validity
             * Check whether the line between the pair of points is 
             * obstacle free 
            **/
            bool get_line_validity(
                Eigen::Vector3d p, Eigen::Vector3d q, Eigen::Vector3d &out)
            {
                // Get the translational difference p to q
                Eigen::Vector3d t_d = q - p;
                // Get the translational vector p to q
                Eigen::Vector3d t_d_pq = t_d.normalized();
                // Get the translational vector q to p
                Eigen::Vector3d t_d_qp = -t_d_pq;
                // Get the translational norm
                double t_n = t_d.norm();

                Eigen::Vector3d p_fd = p;
                Eigen::Vector3d q_fd = q;
                double dist_counter = 0.0;
                double step = global_param.r * 0.9;

                // time_point<std::chrono::system_clock> t_b_t = system_clock::now();

                if (!is_point_within_octree(p_fd) || !is_point_within_octree(q_fd))
                {
                    while (!is_point_within_octree(p_fd))
                    {
                        if (dist_counter > t_n)
                            return true;
                        Eigen::Vector3d vector_step = t_d_pq * step;
                        // Find new p_f
                        p_fd += vector_step;
                        dist_counter += step;
                    }

                    while (!is_point_within_octree(q_fd))
                    {
                        if (dist_counter > t_n)
                            return true;
                        Eigen::Vector3d vector_step = t_d_qp * step;
                        // Find new q_f
                        q_fd += vector_step;
                        dist_counter += step;
                    }

                    if ((q_fd - p_fd).norm() < step)
                        return true;
                }

                // pcl::PointCloud<pcl::PointXYZ>::VectorType voxels_in_line_search;
                // int voxels = (int)_octree.getApproxIntersectedVoxelCentersBySegment(
                //         p_fd, q_fd, voxels_in_line_search, (float)step);
                
                // std::cout << p_fd.transpose() << " - " << 
                //     q_fd.transpose() << std::endl;
                return check_approx_intersection_by_segment(
                    p_fd, q_fd, out);
                
            }
    };
}

#endif