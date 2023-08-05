from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import cv2
import os
import shutil
from moviepy.editor import ImageSequenceClip

def make_grid(grid_size):
    grid = np.full((grid_size, grid_size, 3), (255,255,255), dtype = np.uint8)
    
    # middle obstacle
    for i in range(50):
        for j in range(50):
            grid[i+75][j+80] = (0, 0, 0)

    #parked car 1
    length = 40
    width = 20
    for i in range(width):
        for j in range(length):
            grid[i+170][j+10] = (254, 0, 0)

    # parked car 2
    for i in range(width):
        for j in range(length):
            grid[i+170][j+125] = (254, 0, 0)
    return grid

def rotation_matrix(angle):
    rot_mat = np.array([[(np.cos(angle)), (-np.sin(angle))],
                        [(np.sin(angle)), (np.cos(angle))]])
    return rot_mat

def outline(grid, pos, car_length, car_width):
    grid = copy.deepcopy(grid)
    #pos is a tuple of center pos of the car
    #angle - angle of rotation from the horizontal position
    rot_mat = rotation_matrix(pos[2])
    x_center = int(pos[0] + car_width/2 * np.cos(pos[2]))
    y_center = int(pos[1] + car_width/2 * np.sin(pos[2]))
    pos = np.array([x_center, y_center]).reshape(2,1)
    corners = np.array([[(-car_width/2), ( - car_length/2)],
                        [(- car_width/2), ( + car_length/2)],
                        [(+ car_width/2), (+ car_length/2)],
                        [(+ car_width/2), (- car_length/2)]]).reshape(4,2)
    res_pos = np.dot(rot_mat, corners.T).T
    res_pos = np.array(res_pos, np.int32).reshape(4,2)
    res_pos += np.array([pos[0], pos[1]]).reshape(1,2)
    res_pos = np.vstack((res_pos,[round((res_pos[0][0] + res_pos[1][0])/2) ,round((res_pos[0][1] + res_pos[1][1])/2)]))
    res_pos = np.vstack((res_pos,[round((res_pos[1][0] + res_pos[2][0])/2) ,round((res_pos[1][1] + res_pos[2][1])/2)]))
    res_pos = np.vstack((res_pos,[round((res_pos[2][0] + res_pos[3][0])/2) ,round((res_pos[2][1] + res_pos[3][1])/2)]))
    res_pos = np.vstack((res_pos,[round((res_pos[3][0] + res_pos[0][0])/2) ,round((res_pos[3][1] + res_pos[0][1])/2)]))
    return res_pos

def car_outline(grid, pos, theta_trail, car_length, car_width):
    grid = copy.deepcopy(grid)
    # print(pos)
    #pos is a tuple of center pos of the car
    #angle - angle of rotation from the horizontal position
    rot_mat = rotation_matrix(pos[2])
    x_center = int(pos[0] + car_width/2 * np.cos(pos[2]))
    y_center = int(pos[1] + car_width/2 * np.sin(pos[2]))
    car_pos = np.array([x_center, y_center]).reshape(2,1)
    corners = np.array([[(-car_width/2), ( - car_length/2)],
                        [(- car_width/2), ( + car_length/2)],
                        [(+ car_width/2), (+ car_length/2)],
                        [(+ car_width/2), (- car_length/2)]]).reshape(4,2)
    res_pos = np.dot(rot_mat, corners.T).T
    res_pos = np.array(res_pos, np.int32).reshape(4,2)
    res_pos += np.array([car_pos[0], car_pos[1]]).reshape(1,2)
    for i in range(len(res_pos)):

        cv2.line(grid, (res_pos[0][1],res_pos[0][0] ), (res_pos[1][1], res_pos[1][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos[1][1],res_pos[1][0] ), (res_pos[2][1], res_pos[2][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos[2][1],res_pos[2][0] ), (res_pos[3][1], res_pos[3][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos[3][1],res_pos[3][0] ), (res_pos[0][1], res_pos[0][0]), color = (0,255,0), thickness = 2)
    
    trail_width = 18
    trail_length = 10
    d = 12
    # if theta_trail >=0:
    trail_pos_x = round(pos[0] - d*np.cos(theta_trail))
    trail_pos_y = round(pos[1] - d*np.sin(theta_trail))
    # else:
    #     trail_pos_x = round(pos[0] + d*np.cos(theta_trail))
    #     trail_pos_y = round(pos[1] + d*np.sin(theta_trail))
    rot_mat_trail = rotation_matrix(theta_trail)
    trail_x_center = int(trail_pos_x - trail_width/2 * np.cos(theta_trail))
    trail_y_center = int(trail_pos_y - trail_width/2 * np.sin(theta_trail))
    trail_pos = np.array([trail_x_center, trail_y_center]).reshape(2,1)
    trailer_corners = np.array([[(-trail_width/2), ( - trail_length/2)],
                        [(- trail_width/2), ( + trail_length/2)],
                        [(+ trail_width/2), (+ trail_length/2)],
                        [(+ trail_width/2), (- trail_length/2)]]).reshape(4,2)
    res_pos_trail = np.dot(rot_mat_trail, trailer_corners.T).T
    res_pos_trail = np.array(res_pos_trail, np.int32).reshape(4,2)
    res_pos_trail += np.array([trail_pos[0], trail_pos[1]]).reshape(1,2)

    for i in range(len(res_pos_trail)):

        cv2.line(grid, (res_pos_trail[0][1],res_pos_trail[0][0] ), (res_pos_trail[1][1], res_pos_trail[1][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos_trail[1][1],res_pos_trail[1][0] ), (res_pos_trail[2][1], res_pos_trail[2][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos_trail[2][1],res_pos_trail[2][0] ), (res_pos_trail[3][1], res_pos_trail[3][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos_trail[3][1],res_pos_trail[3][0] ), (res_pos_trail[0][1], res_pos_trail[0][0]), color = (0,255,0), thickness = 2)
    
    cv2.line(grid, (pos[1],pos[0]), (trail_pos_y, trail_pos_x), color = (0,255,0), thickness = 1)
    # plt.imshow(grid, origin='lower')
    # plt.show()
    return grid

def heuristic(current_pos, end_pos):
    return np.sqrt((end_pos[0] - current_pos[0])**2 + (end_pos[1]-current_pos[1])**2) #+ 2 * abs(current_pos[2] - end_pos[2])

def cost(current_pos, start_pos):
    return np.sqrt((start_pos[0] - current_pos[0])**2 + (start_pos[1]-current_pos[1])**2)

class Node():

    def __init__(self, parent_pos = None, pos = None, trail_theta=0):
        self.parent_pos = parent_pos
        self.pos = pos
        self.h = 0
        self.g = 0
        self.f = 0
        self.trail_theta = trail_theta

    def __eq__(self, other):
        return self.pos == other.pos

def planner(grid, start_pos, end_pos, save_path, trail_theta):
    start_node = Node(None, start_pos, trail_theta)
    end_node = Node(None, end_pos, 0)
    sh = grid.shape[0]
    open_list = []
    closed_list = []
    open_list.append(start_node)
    count = 10000
    i = 0

    while(len(open_list) > 0):
        count+=1
        print("count", i, end = "\r")
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):    #find the node with minimum cost out of all childs in open_list
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)
        # print(current_node.pos, current_node.trail_theta)

        if i!=0:
            if (current_node == end_node or current_node.h < 5 or i==10000):
                current = current_node
                path = []
                while current is not start_node:
                    path.append((current.pos, current.trail_theta))
                    current = current.parent_pos
                return path[::-1]
        
        children = []
       
        inputs = [(-1,-30),(-1,-15), (-1,0),(-1,15),(-1,30),\
                (1,-30),(1,-15), (1,0),(1,15),(1,30)]
        L = 20
        d = 12
        dt = 6
        for j, input in enumerate(inputs):
            beta = (input[0]/L)*np.tan(np.deg2rad(input[1]))*dt
            theta_new = (current_node.pos[2]) + beta
            theta_trail_new = (current_node.trail_theta) + (input[0]/d)*np.sin(theta_new - current_node.trail_theta)*dt
            
            x_new = round(current_node.pos[0] + input[0] * np.cos(theta_new)*dt)
            y_new = round(current_node.pos[1] + input[0] * np.sin(theta_new)*dt)

            # check = True
            # for a in range(0,7):
            #     if car_pos[a,0] > sh-1 or car_pos[a,0] < 0 or car_pos[a,1] > sh-1 or car_pos[a,1] < 0:
            #         check = False
            #         break
            #     if grid[car_pos[a][1]][car_pos[a][0]][0] == 0 or grid[car_pos[a][1]][car_pos[a][0]][0] == 254:
            #         check = False
            #         break
            # if check == False:
            #     continue

            node_position = (x_new, y_new, theta_new)
            new_node = Node(current_node, node_position)
            new_node.trail_theta = theta_trail_new
            new_node.h = heuristic(new_node.pos, end_pos)
            new_node.g = current_node.g + cost(new_node.pos, current_node.pos)
            new_node.f = new_node.g + new_node.h
            children.append(new_node)
            

        # while True:pass
        for child in children:
            flag = False
            for closed in closed_list:
                if(closed == child):
                    flag = True
                    break
            for open in open_list:
                if(open == child and child.g >= open.g):
                    flag = True
                    break

            if flag == True:
                continue

            if flag == False:
                open_list.append(child) 
        i += 1

def parking(grid, path, save_path):
    end_pos2 = (150, 155, 0)
    path2 = planner(grid, path[-1][0], end_pos2, save_path, path[-1][1])
    path.extend(path2)
    path.append(((150,155,1.70), 1.90))
    end_pos3 = (160, 120, 0)
    path3 = planner(grid, path[-1][0], end_pos3, save_path, path[-1][1])
    path.extend(path3)
    print(path[-1][0][2])
    end_pos5 = (180, 90, 0)
    path5 = planner(grid, path[-1][0], end_pos5, save_path, path[-1][0][2] - 0.025)
    path.extend(path5)
    path.append(((180,90,1.57), 1.57))

    return path

def path_covered(path, grid, save_path):
    count = 10000
    for i in range(len(path)):
        grid = copy.deepcopy(grid)
        pos = path[i][0]
        theta_trail = path[i][1]
        grid1 = car_outline(grid, pos, theta_trail, 12,25)
        plt.imshow(grid1, origin='lower')
        file_name = save_path + "/" + str(count) + ".png"
        plt.savefig(file_name)
        count += 1
    # grid1 = car_outline(grid, path[-1][0], 0, 12,25)
    plt.imshow(grid1, origin='lower')
    file_name = save_path + "/" + str(count) + ".png"
    plt.savefig(file_name)

def make_video(fps, path, video_file):
    print("Creating video {}, FPS={}".format(video_file, fps))
    clip = ImageSequenceClip(path, fps = fps)
    clip.write_videofile(video_file)
    shutil.rmtree(path)

def main():

    save_path = os.path.join("data")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    else:
        os.mkdir(save_path)

    grid = make_grid(200)
    car_length = 12
    car_width = 25
    start_pos = (20, 25, 0)
    end_pos1 = (130, 60, 0)

    print("1st way point---------------------------")
    path = planner(grid, start_pos, end_pos1, save_path, 0)
    path = parking(grid, path, save_path)
    x = [p[0][0] for p in path]
    y = [p[0][1] for p in path]
    fig = plt.figure()
    plt.plot(y, x)
    fig.suptitle('Planned path for Trailer')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((0,200))
    plt.ylim((0,200))
    plt.savefig('Trailer.png')
    path_covered(path, grid, save_path)
    video_file = 'trailer.mp4'
    make_video(10, save_path, video_file)


if __name__ == "__main__":
    main()