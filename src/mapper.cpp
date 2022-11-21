/*
 * mapper.cpp
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

#define KNRM  "\033[0m"
#define KRED  "\033[31m"
#define KGRN  "\033[32m"
#define KYEL  "\033[33m"
#define KBLU  "\033[34m"
#define KMAG  "\033[35m"
#define KCYN  "\033[36m"
#define KWHT  "\033[37m"

#include <mapper.h>

namespace octree_map
{
    /** 
     * @brief set_parameters
     * Setup the parameters for the mapping module
    **/ 
    void sliding_map::set_parameters(
        double hfov, double vfov, double resolution, 
        double sensor_range, double map_size,
        pcl::PointCloud<pcl::PointXYZ>::Ptr obs_pcl, 
        bool use_sensor, double dist_thres)
    {
        int horizontal_pixels = (int)ceil(
            (sensor_range * tan(hfov/2)) / resolution);
        int vertical_pixels = (int)ceil(
            (sensor_range * tan(vfov/2)) / resolution);

        double vertical_step = 
            vfov / (double)vertical_pixels;
        double horizontal_step = 
            hfov / (double)horizontal_pixels;

        local_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>());

        global_param = {};
        global_param.r = resolution;
        global_param.use_sensor = use_sensor;
        global_param.s_r = sensor_range;
        global_param.radius = map_size/2;
        distance_threshold = dist_thres;

        if (!global_param.use_sensor)
        {
            octree_global.deleteTree();
            // Set the resolution for the octree
            octree_global.setResolution(global_param.r);
            octree_global.setInputCloud(obs_pcl);
            octree_global.addPointsFromInputCloud();
            octree_global.getBoundingBox(
                global_param.mn_b.x(), global_param.mn_b.y(), global_param.mn_b.z(),
                global_param.mx_b.x(), global_param.mx_b.y(), global_param.mx_b.z());
        
            const float min_value = 
                std::numeric_limits<float>::epsilon();

            global_param.m_k[0] =
                (size_t)(std::ceil(
                (global_param.mx_b.x() - global_param.mn_b.x() - min_value) / global_param.r));
            global_param.m_k[1] =
                (size_t)(std::ceil(
                (global_param.mx_b.y() - global_param.mn_b.y() - min_value) / global_param.r));
            global_param.m_k[2] =
                (size_t)(std::ceil(
                (global_param.mx_b.z() - global_param.mn_b.z() - min_value) / global_param.r));
        
            // std::cout << std::ceil((global_param.mx_b.x() - global_param.mn_b.x() - min_value) / global_param.r) << " " 
            //     << std::ceil((global_param.mx_b.y() - global_param.mn_b.y() - min_value) / global_param.r)
            //     << " " << std::ceil((global_param.mx_b.z() - global_param.mn_b.z() - min_value) / global_param.r) << std::endl;

            sensing_offset.clear();
            for (int i = 0; i < vertical_pixels; i++)
                for (int j = 0; j < horizontal_pixels; j++)
                {
                    Eigen::Vector3d q = Eigen::Vector3d(
                        sensor_range * cos(j*horizontal_step - hfov/2.0),
                        sensor_range * sin(j*horizontal_step - hfov/2.0),
                        sensor_range * tan(i*vertical_step - vfov/2.0)
                    );
                    sensing_offset.push_back(q);
                }

        }

        // Set the resolution for the octree
        octree_local.setResolution(global_param.r);
    
        initialized = true;
    }

    /** 
     * @brief check_approx_intersection_by_segment
     * Checks for intersected voxels that contain pointclouds 
     * Edit function from:
     * https://pointclouds.org/documentation/octree__pointcloud_8hpp_source.html#l00269
     * Definition at line 269 of file octree_pointcloud.hpp
    **/ 
    bool sliding_map::check_approx_intersection_by_segment(
        const Eigen::Vector3d origin, const Eigen::Vector3d end, 
        Eigen::Vector3d& intersect)
    {
        pcl::octree::OctreeKey origin_key, end_key;
        pcl::PointXYZ octree_origin(
            origin.x(), origin.y(), origin.z());
        pcl::PointXYZ octree_end(
            end.x(), end.y(), end.z());
        gen_octree_key_for_point(octree_origin, origin_key);
        gen_octree_key_for_point(octree_end, end_key);

        std::vector<Eigen::Vector3i> offset_list;
        
        int extra_safety_volume = 1;
        // Setup the neighbouring boxes
        for (int i = -extra_safety_volume; i <= extra_safety_volume; i++)
            for (int j = -extra_safety_volume; j <= extra_safety_volume; j++)
                for (int k = -extra_safety_volume; k <= extra_safety_volume; k++)
                    offset_list.push_back(Eigen::Vector3i(i, j, k));

        Eigen::Vector3i d(
            end_key.x - origin_key.x,
            end_key.y - origin_key.y,
            end_key.z - origin_key.z
        );

        std::vector<Eigen::Vector3i> idx_list;
        get_bresenham_3d_from_origin(
            d.x(), d.y(), d.z(), idx_list);
        
        for (int i = 0; i < (int)idx_list.size(); i++) 
        {
            pcl::PointXYZ point;
            pcl::octree::OctreeKey query_key(
                origin_key.x + idx_list[i].x(),
                origin_key.y + idx_list[i].y(),
                origin_key.z + idx_list[i].z());
            gen_leaf_node_center_from_octree_key(query_key, point);

            for (Eigen::Vector3i &offset : offset_list)
            {
                pcl::octree::OctreeKey query_point(
                    query_key.x + offset.x(),
                    query_key.y + offset.y(),
                    query_key.z + offset.z()
                );
                
                if (i > 0)
                {
                    if (abs(idx_list[i].x() + offset.x() - idx_list[i-1].x()) <= 1 &&
                    abs(idx_list[i].y() + offset.y() - idx_list[i-1].y()) <= 1 &&
                    abs(idx_list[i].z() + offset.z() - idx_list[i-1].z()) <= 1)
                        continue;
                }

                pcl::PointXYZ neighbour_point;
                gen_leaf_node_center_from_octree_key(
                    query_point, neighbour_point);
                if (octree_global.isVoxelOccupiedAtPoint(neighbour_point))
                {
                    intersect = Eigen::Vector3d(
                        point.x, point.y, point.z);
                    return false;
                }
            }
        }
        
        return true;
    }

    /** 
     * @brief gen_leaf_node_center_from_octree_key
     * Edited from the protected function for octree
     * void pcl::octree::OctreePointCloud<PointT, LeafContainerT, BranchContainerT, OctreeT>::
     * genLeafNodeCenterFromOctreeKey(const OctreeKey& key, PointT& point) const
    **/
    void sliding_map::gen_leaf_node_center_from_octree_key(
        const pcl::octree::OctreeKey key, pcl::PointXYZ& point)
    {
        // define point to leaf node voxel center
        point.x = static_cast<float>(
            (static_cast<double>(key.x) + 0.5f) * 
            global_param.r + global_param.mn_b.x());
        point.y = static_cast<float>(
            (static_cast<double>(key.y) + 0.5f) * 
            global_param.r + global_param.mn_b.y());
        point.z =static_cast<float>(
            (static_cast<double>(key.z) + 0.5f) * 
            global_param.r + global_param.mn_b.z());
    }

    /** 
     * @brief gen_octree_key_for_point
     * Edited from the protected function for octree
     * void pcl::octree::OctreePointCloud<PointT, LeafContainerT, BranchContainerT, OctreeT>::
     * genOctreeKeyforPoint(const PointT& point_arg, OctreeKey& key_arg) const
    **/
    void sliding_map::gen_octree_key_for_point(
        const pcl::PointXYZ point_arg, 
        pcl::octree::OctreeKey& key_arg)
    {
        // calculate integer key for point coordinates
        key_arg.x = static_cast<uint8_t>((point_arg.x - global_param.mn_b.x()) / global_param.r);
        key_arg.y = static_cast<uint8_t>((point_arg.y - global_param.mn_b.y()) / global_param.r);
        key_arg.z = static_cast<uint8_t>((point_arg.z - global_param.mn_b.z()) / global_param.r);

        // std::cout << global_param.m_k[0] << " " << global_param.m_k[1] 
        //     << " " << global_param.m_k[2] << std::endl;
        // std::cout << key_arg.x << " " << key_arg.y << " " << key_arg.z << std::endl;

        assert(key_arg.x <= global_param.m_k[0]);
        assert(key_arg.y <= global_param.m_k[1]);
        assert(key_arg.z <= global_param.m_k[2]);
    }


    /** 
     * @brief get_raycast_on_pcl
    **/
    pcl::PointCloud<pcl::PointXYZ>::Ptr 
        sliding_map::get_raycast_on_pcl(
        Eigen::Vector3d p, Eigen::Quaterniond q)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr 
            tmp(new pcl::PointCloud<pcl::PointXYZ>);

        for (int i = 0; i < (int)sensing_offset.size(); i++)
        {
            Eigen::Quaterniond point;
            point.w() = 0;
            point.vec() = sensing_offset[i];
            Eigen::Quaterniond rotatedP = q * point * q.inverse(); 
            Eigen::Vector3d x = p + rotatedP.vec();

            Eigen::Vector3d intersect;
            if (!get_line_validity(
                p, x, intersect))
            {
                // Eigen::Vector3d direction = (x - p).normalized(); 
                // intersect += m_p.s_m_r/2 * direction;
                pcl::PointXYZ add;
                add.x = intersect.x();
                add.y = intersect.y();
                add.z = intersect.z();
                tmp->points.push_back(add);
            }

        }

        return tmp;
    }

    /** 
     * @brief get_sliding_map_from_global
     * If there is a global map present, use this
    **/
    pcl::PointCloud<pcl::PointXYZ>::Ptr 
        sliding_map::get_sliding_map_from_global(
        Eigen::Vector3d p, Eigen::Quaterniond q)
    {
        time_point<std::chrono::system_clock> ray_timer = system_clock::now();
        pcl::PointCloud<pcl::PointXYZ>::Ptr 
            local_cloud_current = get_raycast_on_pcl(p, q);
        double ray_time = duration<double>(system_clock::now() - ray_timer).count();
        std::cout << "raycast time (" << KBLU << ray_time * 1000 << KNRM << "ms)" << std::endl;

        if (!local_cloud_current->points.empty())
        {
            *local_cloud_current += *local_cloud;

            octree_local.deleteTree();
            octree_local.setInputCloud(local_cloud_current);
            octree_local.addPointsFromInputCloud();
        }

        occupied_size = 
            octree_local.getOccupiedVoxelCenters(occupied_voxels);
        
        Eigen::Vector3d voxel_center;
        get_estimated_center_of_point(p, voxel_center);

        extract_point_cloud_within_boundary(p);
        
        update_pos_vector(p);

        double extraction_time = duration<double>(system_clock::now() - ray_timer).count() - ray_time;
        std::cout << "extraction time (" << KBLU << extraction_time * 1000 << KNRM << "ms)" << std::endl;

        double ray_n_acc_time = duration<double>(system_clock::now() - ray_timer).count();
        std::cout << "raycast and accumulation time (" << KBLU << ray_n_acc_time * 1000 << KNRM << "ms)" << std::endl;

        return local_cloud;
    }

    /** 
     * @brief get_sliding_map_from_sensor
     * If there is only sensor data present, use this
    **/
    pcl::PointCloud<pcl::PointXYZ>::Ptr 
        sliding_map::get_sliding_map_from_sensor(
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl,    
        Eigen::Vector3d p, Eigen::Quaterniond q)
    {
        if (!pcl->points.empty())
        {
            *pcl += *local_cloud;

            octree_local.deleteTree();
            octree_local.setInputCloud(pcl);
            octree_local.addPointsFromInputCloud();
        }

        occupied_size = 
            octree_local.getOccupiedVoxelCenters(occupied_voxels);
        
        Eigen::Vector3d voxel_center;
        get_estimated_center_of_point(p, voxel_center);

        extract_point_cloud_within_boundary(p);

        update_pos_vector(p);

        return local_cloud;
    }

    void sliding_map::get_estimated_center_of_point(
        Eigen::Vector3d p, Eigen::Vector3d &est)
    {
        Eigen::Vector3d dist = (global_param.mn_b - p);
        // Eigen::Vector3d v = (global_param.mn_b - p).normalized();
        int nx = (int)round(abs(dist.x())/global_param.r);
        int ny = (int)round(abs(dist.y())/global_param.r);
        int nz = (int)round(abs(dist.z())/global_param.r);
        est = Eigen::Vector3d(
            (nx + 0.5f)*global_param.r + global_param.mn_b.x(),
            (ny + 0.5f)*global_param.r + global_param.mn_b.y(),
            (nz + 0.5f)*global_param.r + global_param.mn_b.z()
        );
    }

    void sliding_map::extract_point_cloud_within_boundary(
        Eigen::Vector3d c)
    {
        if (!local_cloud->points.empty())
            local_cloud->points.clear();
        
        if (occupied_size == 0)
            return;

        double rr = pow(global_param.radius, 2);
        for (auto &v : occupied_voxels)
        {
            double xx = pow(v.x - c.x(), 2);
            double yy = pow(v.y - c.y(), 2);
            double zz = pow(v.z - c.z(), 2);
            if (xx + yy + zz < rr)
                local_cloud->points.push_back(v);
        }
    }

    /**
     * @brief get_bresenham_3d_from_origin
     * Modified from:
     * https://gist.github.com/yamamushi/5823518
     */
    void sliding_map::get_bresenham_3d_from_origin( 
        const int x2, const int y2, const int z2, 
        std::vector<Eigen::Vector3i> &idx)
    {
        idx.clear();

        int i, dx, dy, dz, l, m, n, x_inc, y_inc, z_inc, err_1, err_2, dx2, dy2, dz2;
        Eigen::Vector3i point(0, 0, 0);

        dx = x2 - 0;
        dy = y2 - 0;
        dz = z2 - 0;
        x_inc = (dx < 0) ? -1 : 1;
        l = abs(dx);
        y_inc = (dy < 0) ? -1 : 1;
        m = abs(dy);
        z_inc = (dz < 0) ? -1 : 1;
        n = abs(dz);
        dx2 = l << 1;
        dy2 = m << 1;
        dz2 = n << 1;
        
        if ((l >= m) && (l >= n)) {
            err_1 = dy2 - l;
            err_2 = dz2 - l;
            for (i = 0; i < l; i++) 
            {
                idx.push_back(point);
                if (err_1 > 0) {
                    point[1] += y_inc;
                    err_1 -= dx2;
                }
                if (err_2 > 0) {
                    point[2] += z_inc;
                    err_2 -= dx2;
                }
                err_1 += dy2;
                err_2 += dz2;
                point[0] += x_inc;
            }
        } else if ((m >= l) && (m >= n)) {
            err_1 = dx2 - m;
            err_2 = dz2 - m;
            for (i = 0; i < m; i++) {
                idx.push_back(point);
                if (err_1 > 0) {
                    point[0] += x_inc;
                    err_1 -= dy2;
                }
                if (err_2 > 0) {
                    point[2] += z_inc;
                    err_2 -= dy2;
                }
                err_1 += dx2;
                err_2 += dz2;
                point[1] += y_inc;
            }
        } else {
            err_1 = dy2 - n;
            err_2 = dx2 - n;
            for (i = 0; i < n; i++) {
                idx.push_back(point);
                if (err_1 > 0) {
                    point[1] += y_inc;
                    err_1 -= dz2;
                }
                if (err_2 > 0) {
                    point[0] += x_inc;
                    err_2 -= dz2;
                }
                err_1 += dy2;
                err_2 += dx2;
                point[2] += z_inc;
            }
        }
        idx.push_back(point);
    }
}