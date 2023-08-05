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
            grid[i+170][j+20] = (254, 0, 0)

    #parked car 2
    for i in range(width):
        for j in range(length):
            grid[i+170][j+110] = (254, 0, 0)
    return grid

def rotation_matrix(angle):
    rot_mat = np.array([[(np.cos(angle)), (-np.sin(angle))],
                        [(np.sin(angle)), (np.cos(angle))]])
    return rot_mat

def outline(grid, pos, angle, car_length, car_width):
    grid = copy.deepcopy(grid)
    #pos is a tuple of center pos of the car
    #angle - angle of rotation from the horizontal position
    rot_mat = rotation_matrix(angle)
    pos = np.array(pos).reshape(2,1)
    corners = np.array([[(-car_width/2), ( - car_length/2)],
                        [(- car_width/2), ( + car_length/2)],
                        [(+ car_width/2), (+ car_length/2)],
                        [(+ car_width/2), (- car_length/2)]]).reshape(4,2)
    res_pos = np.dot(rot_mat, corners.T).T
    res_pos = np.array(res_pos, np.int32).reshape(4,2)
    res_pos += np.array([pos[0], pos[1]]).reshape(1,2)

    return res_pos

def car_outline(grid, pos, angle, car_length, car_width):
    grid = copy.deepcopy(grid)
    #pos is a tuple of center pos of the car
    #angle - angle of rotation from the horizontal position
    rot_mat = rotation_matrix(angle)
    pos = np.array(pos).reshape(2,1)
    corners = np.array([[(-car_width/2), ( - car_length/2)],
                        [(- car_width/2), ( + car_length/2)],
                        [(+ car_width/2), (+ car_length/2)],
                        [(+ car_width/2), (- car_length/2)]]).reshape(4,2)
    res_pos = np.dot(rot_mat, corners.T).T
    res_pos = np.array(res_pos, np.int32).reshape(4,2)
    res_pos += np.array([pos[0], pos[1]]).reshape(1,2)
    for i in range(len(res_pos)):
        cv2.line(grid, (res_pos[0][1],res_pos[0][0] ), (res_pos[1][1], res_pos[1][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos[1][1],res_pos[1][0] ), (res_pos[2][1], res_pos[2][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos[2][1],res_pos[2][0] ), (res_pos[3][1], res_pos[3][0]), color = (0,255,0), thickness = 2)
        cv2.line(grid, (res_pos[3][1],res_pos[3][0] ), (res_pos[0][1], res_pos[0][0]), color = (0,255,0), thickness = 2)

    # plt.imshow(grid)
    # plt.show()
    return grid

def heuristic(current_pos, end_pos, theta):
    return np.sqrt((end_pos[0] - current_pos[0])**2 + (end_pos[1]-current_pos[1])**2)

def cost(current_pos, start_pos, theta):
    return np.sqrt((start_pos[0] - current_pos[0])**2 + (start_pos[1]-current_pos[1])**2)

class Node():

    def __init__(self, parent_pos = None, pos = None):
        self.parent_pos = parent_pos
        self.pos = pos
        self.h = 0
        self.g = 0
        self.f = 0
        self.theta = 0
        # self.R = 5    #radius of wheel
        # self.l = 50    #distance of each wheel from center of axle

    def __eq__(self, other):
        return self.pos == other.pos

def planner(grid, start_pos, end_pos, save_path):
    start_node = Node(None, start_pos)
    end_node = Node(None, end_pos)
    sh = grid.shape[0]
    open_list = []
    closed_list = []
    grid = cv2.circle(grid, (end_node.pos[1],end_node.pos[0]), radius =2, color =(255, 0, 0))
    open_list.append(start_node)
    count = 10000
    i = 0
    path_count = 0
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

        if current_node == end_node:
            current = current_node
            path = []
            while current is not None:
                path.append((current.pos, current.theta))
                current = current.parent_pos
            return path[::-1]
        
        children = []
        r = 5
        l = 12
        ur = [-1,0,1]
        ul = [-1,0,1]
        inputs = [(-1,-1), (-1,0), (-1,1), (0,1),(1,1), (1,0), (1,-1), (0,-1)]
        dt = 1
        for input in inputs:
            
            theta_new = current_node.theta + (r/l)*(input[0] - input[1])*dt
            x_new = round(current_node.pos[0] + (r/2) * np.cos(theta_new)*2*dt)
            y_new = round(current_node.pos[1] + (r/2) * np.sin(theta_new)*2*dt)

            # if x_new > sh-1 or x_new < 0 or y_new > sh-1 or y_new < 0:
            #     continue
            # if grid[x_new][y_new][0] == 0 or grid[x_new][y_new][0] == 254:
            #     continue

            node_position = (x_new, y_new)
            new_node = Node(current_node, node_position)
            new_node.h = heuristic(new_node.pos, end_pos, theta_new)
            new_node.g = current_node.g + cost(new_node.pos, current_node.pos, theta_new)
            new_node.f = new_node.g + new_node.h
            new_node.theta = theta_new
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

def path_covered(path, grid, save_path, car_length, car_width):
    count = 10000
    for i in range(len(path)):
        grid = copy.deepcopy(grid)
        pos = path[i][0]
        angle = path[i][1]
        grid1 = car_outline(grid, pos, angle, car_length, car_width)
        plt.imshow(grid1, origin='lower')
        file_name = save_path + "/" + str(count) + ".png"
        plt.savefig(file_name)
        count += 1
    grid1 = car_outline(grid, path[-1][0], 1.57, car_length, car_width)
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
    start_pos = (20,25)
    end_pos = (180, 85)
    
    path = planner(grid, start_pos, end_pos, save_path)
    x = [p[0][0] for p in path]
    y = [p[0][1] for p in path]
    fig = plt.figure()
    plt.plot(y, x)
    fig.suptitle('Planned path for Diwheel Robot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim((0,200))
    plt.ylim((0,200))
    plt.savefig('DiffDrive.png')
    path_covered(path, grid, save_path, car_length, car_width)
    video_file = 'diffdrive.mp4'
    make_video(10, save_path, video_file)


if __name__ == "__main__":
    main()