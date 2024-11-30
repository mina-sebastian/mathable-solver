import os
import cv2 as cv
import numpy as np
from extractor import extrage_careu, show_image, getLines, get_patches
from config import box_size, boxes, total_size, game_board, starting_points, initial_signs

FOLDER_TRAIN = "./antrenare"
FOLDER_SCORES = "./fisiere_solutie/342_Chirus_Mina_Sebastian/"

def format_chess_like_position(y, x):
    return f"{y + 1}{chr(ord('A') + (x))}"

def write_score_file(round, points):
    file_path = os.path.join(FOLDER_SCORES, f"{round}_scores.txt")
    with open(file_path, "w") as file:
        file.write(f"{points}")

def write_annotation_file(round, turn, current_pos, current_token):
    if(turn < 10):
        name_turn = '0'+str(turn)
    else:
        name_turn = str(turn)
    file_path = os.path.join(FOLDER_SCORES, f"{round}_{name_turn}.txt")
    with open(file_path, "w") as file:
        file.write(f"{current_pos} {current_token}")
    

def init_game():
    print("-*-"*10)
    print("SETTING UP FOLDERS...")
    print("-*-"*10)

    if not os.path.exists(FOLDER_TRAIN):
        print("NU EXISTA FOLDERUL DE TRAINING", FOLDER_TRAIN)
        os.makedirs(FOLDER_TRAIN)
    
    print("-*-"*10)
    print("DONE: SETTING UP FOLDERS")
    print("-*-"*10)


    if not os.path.exists(FOLDER_SCORES):
        os.makedirs(FOLDER_SCORES)

def get_possible_points(nimg, current_board, current_positions):
    # for current_position in current_positions:
    #     print(game_board[current_position[0]][current_position[1]])
    possible_points = []
    # print('get_possible_points')
    # print(current_positions)
    # print(current_board)
    for current_position in current_positions:
        y, x = current_position
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if abs(i) == abs(j):
                    continue
                new_x = x + i
                new_y = y + j
                if (new_y, new_x) in starting_points:
                    continue
                if (new_y, new_x) in current_positions:
                    continue
                if new_x < 0 or new_x >= boxes or new_y < 0 or new_y >= boxes:
                    continue
                if current_board[new_y][new_x] != '' and current_board[new_y][new_x] not in initial_signs:
                    continue
                prev_x = x - i
                prev_y = y - j
                if prev_x < 0 or prev_x >= boxes or prev_y < 0 or prev_y >= boxes:
                    continue
                # img = nimg.copy()
                # print('-'*100)
                # print(initial_signs)
                # print('prev', prev_x, prev_y, ':', current_board[prev_y][prev_x], current_board[prev_y][prev_x] in initial_signs, current_board[prev_y][prev_x] == '')
                # print('current', x, y, ':', current_board[y][x])
                # print('new', new_x, new_y, ':', current_board[new_y][new_x])
                # cv.circle(img, (x * box_size + box_size // 2, y * box_size + box_size // 2), 10, (0, 0, 255), -1)
                # cv.circle(img, (prev_x * box_size + box_size // 2, prev_y * box_size + box_size // 2), 10, (255, 255, 0), -1)
                # cv.circle(img, (new_x * box_size + box_size // 2, new_y * box_size + box_size // 2), 10, (255, 0, 0), -1)
                if (current_board[prev_y][prev_x] in initial_signs and not current_board[prev_y][prev_x].isnumeric()) or (current_board[prev_y][prev_x] == ''):
                    continue
                # show_image('img', img)
                # print('dir', new_x, new_y, '-', current_board[new_y][new_x])
                # print(prev_x, prev_y, '-' ,current_board[prev_y][prev_x])

                possible_points.append((new_y, new_x))
    return possible_points


def get_files(fast=False):
    print("-*-"*10)
    print("GETTING FILES")
    print("-*-"*10)
    rez = {}
    for file_name in sorted(os.listdir(FOLDER_TRAIN)):
        round, turn = file_name.split("_")
        round = int(round)
        turn_int = int(turn.split(".")[0]) if (turn.split(".")[0]).isdigit() else 0
        # if turn_int > 3:
        #     continue
        # if fast and turn.startswith('10'):
        #     break
        print(round, turn)
        file_path = os.path.join(FOLDER_TRAIN, file_name)
        if round not in rez:
            rez[round] = {
                "paths": [],
                "turns": ''
            }
        if turn.endswith("turns.txt"):
            file_content = open(file_path, "r").readlines()
            rez[round]["turns"] = [line.strip() for line in file_content]
        elif turn.endswith(".jpg"):
            # img = cv.imread(file_path)
            rez[round]["paths"].append(file_path)
    print("-*-"*10)
    print("DONE: FILES READ")
    print("-*-"*10)
    return rez


lower_patch = np.array([0, 162, 180])
upper_patch = np.array([214, 255, 231])

# lower_patch = np.array([0, 0, 152])
# upper_patch = np.array([255, 255, 255])

init_game()
files = get_files(True)
for round in range(1, 5):
    # print(files[str(i)])
    prev_patches = None
    prev_board = None
    current_board = game_board.copy()
    current_points = starting_points.copy()
    turn = 1
    for img_path in files[round]["paths"]:
        img = cv.imread(img_path)
        res, res_sharp = extrage_careu(img)
        lines_v, lines_h = getLines(res)
        # for line in lines_vertical:
        #         cv.line(result, line[0], line[1], (0, 255, 0), 5)
        #     for line in lines_horizontal:
        #         cv.line(result, line[0], line[1], (0, 0, 255), 5)
        # show_image('lines_h', lines_h)
        # show_image('lines_v', lines_v)
        patches = get_patches(res, lines_v, lines_h)

        rez = np.zeros((total_size, total_size, 3), dtype=np.uint8)
        for i in range(boxes):
            for j in range(boxes):
                x = i * box_size
                y = j * box_size
                # show_image('img', patches[i][j])
                rez[y:y+box_size, x:x+box_size] = patches[j][i]
        
        # for i in range(boxes):
        #     for j in range(boxes):
        #         x = i * box_size
        #         y = j * box_size
        #         if game_board[j][i] != '':
        #             cv.putText(rez, game_board[j][i], (x, y+60), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
        # for starting_point in starting_points:
        #     y, x = starting_point
        #     cv.circle(rez, (x * box_size + box_size // 2, y * box_size + box_size // 2), 10, (0, 0, 255), -1)
        # for possible_point in get_possible_points(current_board, starting_points):
        #     y, x = possible_point
        #     cv.circle(rez, (x * box_size + box_size // 2, y * box_size + box_size // 2), 10, (0, 255, 0), -1)
        if prev_board is None:
            possible_points = get_possible_points(res, current_board, current_points)
            possible_patches = [patches[y][x] for y, x in possible_points]
            possible_patches = [cv.inRange(patch, lower_patch, upper_patch) for patch in possible_patches]
            greatest_mean = 0
            chosen = None
            chosen_x, chosen_y = None, None
            # print(possible_points)
            for y, x in possible_points:
                patch = patches[y][x]
                patch = cv.inRange(patch, lower_patch, upper_patch)
                patch_mean = np.mean(patch)
                if patch_mean > greatest_mean:
                    greatest_mean = patch_mean
                    chosen = patch
                    chosen_x, chosen_y = x, y
            if chosen is not None:
                current_board[chosen_y][chosen_x] = '1'
                current_points.append((chosen_y, chosen_x))
                write_annotation_file(round, turn, format_chess_like_position(chosen_y, chosen_x), '1')

            prev_patches = patches
            prev_board = current_board.copy()
        else:
            for current_position in current_points:
                y, x = current_position
                cv.circle(rez, (x * box_size + box_size // 2, y * box_size + box_size // 2), 10, (0, 0, 255), -1)
            possible_points = get_possible_points(res, current_board, current_points)
            for possible_point in possible_points:
                y, x = possible_point
                cv.circle(rez, (x * box_size + box_size // 2, y * box_size + box_size // 2), 10, (0, 255, 0), -1)
            # show_image('img_show_all', rez)
            biggest_diff = 0
            chosen_patch = None
            chosen_x, chosen_y = None, None

            for y, x in possible_points:
                current_patch = patches[y][x]
                prev_patch = prev_patches[y][x]
                diff = cv.absdiff(current_patch, prev_patch)
                if np.sum(diff) > biggest_diff:
                    # print(np.sum(diff))
                    biggest_diff = np.sum(diff)
                    chosen_x, chosen_y = x, y
                    chosen_patch = current_patch
                # show_image('img', diff)
            if chosen_patch is not None:
                current_board[chosen_y][chosen_x] = '1'
                current_points.append((chosen_y, chosen_x))
                # show_image('img', chosen_patch)
                show_where = res.copy()
                cv.circle(show_where, (chosen_x * box_size + box_size // 2, chosen_y * box_size + box_size // 2), 10, (0, 0, 255), -1)
                # show_image('where', show_where)
                write_annotation_file(round, turn, format_chess_like_position(chosen_y, chosen_x), '1')

            prev_board = current_board.copy()
            prev_patches = patches

        # show_image('img', rez)
        turn += 1
print(files)

