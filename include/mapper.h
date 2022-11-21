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

// Convex Decomposition includes
// #include <decomp_util/ellipsoid_decomp.h>

// using namespace Eigen;
// using namespace std;
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
    namespace convex{
        #include <decomp_util/line_segment.h>
    }

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

            struct triangles 
            {
                std::vector<Eigen::Vector3i> tri_idx;
                std::vector<Eigen::Vector3d> vert;
            };

            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree_global = 
                decltype(octree_global)(0.1);
            pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree_local = 
                decltype(octree_local)(0.1);

            pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud;

            pcl::PointCloud<pcl::PointXYZ>::VectorType occupied_voxels;

            map_parameters global_param;

            int occupied_size = 0;
            
            double distance_threshold;

            bool initialized;
            bool save_polygon;

            std::vector<Eigen::Vector3d> key_pos;
            std::vector<Eigen::Vector3d> sensing_offset;

            convex::vec_E<convex::Polyhedron<3>> poly_safe;

            std::mutex update_pose_mutex;

            std::vector<triangles> safe_tri_vector;

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
                bool use_sensor, double dist_thres);

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

            std::vector<triangles> visualize_safe_corridor()
            {
                using namespace convex;

                std::lock_guard<std::mutex> pose_lock(
                    update_pose_mutex);

                // time_point<std::chrono::system_clock> polygon_timer = system_clock::now();

                std::vector<triangles> tri_vector;

                if (key_pos.size() < 2)
                    return tri_vector;

                if (save_polygon && key_pos.size() >= 3)
                {
                    Polyhedron3D poly = get_polyhedron_from_line(
                        std::make_pair(key_pos[key_pos.size()-3], key_pos[key_pos.size()-2]));

                    vec_E<vec_Vec3f> vert = 
                        get_vertices_from_polygons(poly);
                    
                    safe_tri_vector.push_back(
                        get_triangles_of_polygon(vert));
                    
                    save_polygon = false;
                }

                for (size_t i = 0; i < key_pos.size()-1; i++)
                {
                    Polyhedron3D poly = get_polyhedron_from_line(
                        std::make_pair(key_pos[i], key_pos[i+1]));

                    vec_E<vec_Vec3f> vert = 
                        get_vertices_from_polygons(poly);
                    
                    tri_vector.push_back(
                        get_triangles_of_polygon(vert));
                }

                Polyhedron3D poly = get_polyhedron_from_line(
                    std::make_pair(key_pos[key_pos.size()-2], key_pos[key_pos.size()-1]));

                vec_E<vec_Vec3f> vert = 
                    get_vertices_from_polygons(poly);
                
                tri_vector.push_back(
                    get_triangles_of_polygon(vert));

                for (triangles &tri : safe_tri_vector)
                    tri_vector.push_back(tri);

                // double polygon_time = duration<double>(system_clock::now() - polygon_timer).count();
                // std::cout << "polygon time (" << KBLU << polygon_time * 1000 << KNRM << "ms) cloud size (" <<
                //     occupied_size << ")" << std::endl;

                return tri_vector;
            }

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

            void update_pos_vector(Eigen::Vector3d &p)
            {
                std::lock_guard<std::mutex> pose_lock(
                    update_pose_mutex);
                
                if (key_pos.empty())
                {
                    key_pos.push_back(p);
                    key_pos.push_back(p);
                    return;
                }

                double distance_travelled = 
                    (p - key_pos[key_pos.size()-2]).norm();
                
                if (distance_travelled > distance_threshold)
                {
                    key_pos.erase(key_pos.end());
                    key_pos.push_back(p);
                    key_pos.push_back(p);
                    save_polygon = true;
                }
                else
                {
                    key_pos.erase(key_pos.end());
                    key_pos.push_back(p);
                }
                
                int remove_index = -1;
                for (size_t i = key_pos.size()-1; i >= 0; i--)
                {
                    double xx = pow(key_pos[i].x() - p.x(), 2);
                    double yy = pow(key_pos[i].y() - p.y(), 2);
                    double zz = pow(key_pos[i].z() - p.z(), 2);
                    if (xx + yy + zz > pow(global_param.radius,2))
                    {
                        remove_index = (int)i;
                        break;
                    }
                }
                if (remove_index != -1)
                {
                    for (int i = 0; i <= remove_index; i++)
                        key_pos.erase(key_pos.begin());
                }
            };

            convex::Polyhedron3D 
                get_polyhedron_from_line(
                std::pair<Eigen::Vector3d, Eigen::Vector3d> line)
            {
                using namespace convex;

                vec_Vec3f obs_vec; // Vector that contains the occupied points
                
                // Eigen::Vector3d mid = (line.first + line.second) / 2.0;
                // // inflated radius
                // double radius = (line.first - line.second).norm() * 1.5;

                // std::vector<int> pointIdxRadiusSearch;
                // std::vector<float> pointRadiusSquaredDistance;

                // if (octree_local.radiusSearch(
                //     pcl::PointXYZ(mid.x(), mid.y(), mid.z()), 
                //     radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
                // {
                //     for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
                //     {
                //         obs_vec.push_back(Vec3f(
                //             (*local_cloud)[pointIdxRadiusSearch[i]].x, 
                //             (*local_cloud)[pointIdxRadiusSearch[i]].y, 
                //             (*local_cloud)[pointIdxRadiusSearch[i]].z));
                        
                //     }
                // }

                for (pcl::PointXYZ &p : local_cloud->points)
                    obs_vec.push_back(Vec3f(p.x, p.y, p.z));

                Vec3f pos1(
                    line.first.x(), line.first.y(), line.first.z());
                Vec3f pos2(
                    line.second.x(), line.second.y(), line.second.z());

                LineSegment3D decomp(pos1, pos2);

                decomp.set_obs(obs_vec);
                // Only try to find cvx decomp in the Minkowski sum of JPS and
                // this box (I think) par_.drone_radius
                decomp.set_local_bbox(Vec3f(2.5, 2.5, 1.5));
                decomp.dilate(0);

                return decomp.get_polyhedron();
            }

            /**
             * @brief get_vertices_from_polygons
             * https://github.com/sikang/DecompROS cal_vertices
             * @param poly 
             * @return vec_E<vec_Vec3f> 
             **/
            inline convex::vec_E<convex::vec_Vec3f> 
                get_vertices_from_polygons(
                const convex::Polyhedron3D &poly) 
            {
                using namespace convex;
                
                vec_E<vec_Vec3f> bds;
                const auto vts = poly.hyperplanes();
                
                // for each plane, find lines on it
                for (unsigned int i = 0; i < vts.size(); i++) 
                {
                    const Vec3f t = vts[i].p_;
                    const Vec3f n = vts[i].n_;
                    const Quatf q = Quatf::FromTwoVectors(Vec3f(0, 0, 1), n);
                    const Mat3f R(q); // body to world
                    vec_E<std::pair<Vec2f, Vec2f>> lines;
                    for (unsigned int j = 0; j < vts.size(); j++) {
                    if (j == i)
                        continue;
                    Vec3f nw = vts[j].n_;
                    Vec3f nb = R.transpose() * nw;
                    decimal_t bb = vts[j].p_.dot(nw) - nw.dot(t);
                    Vec2f v = Vec3f(0, 0, 1).cross(nb).topRows<2>(); // line direction
                    Vec2f p;                                         // point on the line
                    if (nb(1) != 0)
                        p << 0, bb / nb(1);
                    else if (nb(0) != 0)
                        p << bb / nb(0), 0;
                    else
                        continue;
                    lines.push_back(std::make_pair(v, p));
                    }

                    // find all intersect points
                    vec_Vec2f pts = line_intersects(lines);
                    
                    // filter out points inside polytope
                    vec_Vec2f pts_inside;
                    for (const auto &it : pts) {
                    Vec3f p = R * Vec3f(it(0), it(1), 0) + t; // convert to world frame
                    if (poly.inside(p))
                        pts_inside.push_back(it);
                    }

                    if (pts_inside.size() > 2) {
                    // sort in plane frame
                    pts_inside = sort_pts(pts_inside);

                    // transform to world frame
                    vec_Vec3f points_valid;
                    for (auto &it : pts_inside)
                        points_valid.push_back(R * Vec3f(it(0), it(1), 0) + t);

                    // insert resulting polygon
                    bds.push_back(points_valid);
                    }
                }

                return bds;
            }

            /**
             * @brief get_triangles_of_polygon
             * https://github.com/sikang/DecompROS setMessage
             * @param bds 
             * @return triangles
             **/
            triangles get_triangles_of_polygon(
                const convex::vec_E<convex::vec_Vec3f> &bds) 
            {
                using namespace convex;

                triangles t;

                if (bds.empty())
                    return t;

                int free_cnt = 0;
                for (const auto &vs: bds) 
                {
                    if (vs.size() > 2) 
                    {
                        Vec3f p0 = vs[0];
                        Vec3f p1 = vs[1];
                        Vec3f p2 = vs[2];
                        Vec3f n = (p2-p0).cross(p1-p0);
                        n = n.normalized();
                        if(std::isnan(n(0)))
                        n = Vec3f(0, 0, -1);

                        int ref_cnt = free_cnt;
                        // Ogre::Vector3 normal(n(0), n(1), n(2));
                        for (unsigned int i = 0; i < vs.size(); i++) 
                        {
                            // obj_->addVertex(Ogre::Vector3(vs[i](0), vs[i](1), vs[i](2)), normal);
                            t.vert.push_back(
                                Eigen::Vector3d(vs[i](0), vs[i](1), vs[i](2)));
                            if (i > 1 && i < vs.size())
                                t.tri_idx.push_back(
                                    Eigen::Vector3i(ref_cnt, free_cnt - 1, free_cnt));
                                // obj_->addTriangle(ref_cnt, free_cnt - 1, free_cnt);
                            free_cnt++;
                        }
                    }
                }

                return t;
            }

    };
}

#endif