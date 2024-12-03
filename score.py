from copy import deepcopy
import math
import os
import cv2 as cv
import numpy as np
from extractor import extrage_careu, show_image, getLines, get_patches
from config import box_size, boxes, total_size, game_board as initial_game_board, starting_points, initial_signs, pieces, numbers, ROUNDS_FIRST, ROUNDS_LAST, FOLDER_IMAGES, FOLDER_TEMPLATES, FOLDER_SCORES



templates = {}

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
    
# def show_in_grid():
#     total_size_x = 0
#     total_size_y = 0
#     for nb in numbers:
#         total_size_x += templates[nb].shape[1]
#         total_size_y += templates[nb].shape[0]
#     collage = np.zeros((total_size_y, total_size_x, 3), dtype=np.uint8)
#     x = 0
#     y = 0
#     for nb in numbers:
#         collage[y:y+templates[nb].shape[0], x:x+templates[nb].shape[1]] = templates[nb]
#         x += templates[nb].shape[1]
#         if x >= total_size_x:
#             x = 0
#             y += 90
#     find_color_values_using_trackbar(collage)
#     show_image('collage', collage)

def sharp_n_thresh(img):
    image_m_blur = cv.medianBlur(img,3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    _, thresh = cv.threshold(image_sharpened, 0, 255, cv.THRESH_BINARY)
    thresh = cv.bitwise_not(thresh)
    return thresh

def init_game():
    print("-*-"*10)
    print("SETTING UP FOLDERS...")
    print("-*-"*10)

    if not os.path.exists(FOLDER_IMAGES):
        print("NU EXISTA FOLDERUL CU IMAGINI", FOLDER_IMAGES)
        exit(1)
    if not os.path.exists(FOLDER_TEMPLATES):
        print("NU EXISTA FOLDERUL CU TEMPLATES", FOLDER_TEMPLATES)
        exit(1)

    if not os.path.exists(FOLDER_SCORES):
        os.makedirs(FOLDER_SCORES)
    
    for i in pieces.keys():
        templates[i] = {}

    for file_name in sorted(os.listdir(FOLDER_TEMPLATES)):
        if file_name.startswith('a') and file_name.endswith(".jpg"):
            number = int(file_name.replace('a', '').replace('.jpg', ''))
            templates[number] = []
            img = cv.imread(os.path.join(FOLDER_TEMPLATES, file_name))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            thresh = sharp_n_thresh(img)
            for deg in range(-15, 16, 2):
                for erosion in range(-2, 3):
                    for scale in [0.8, 0.9, 1, 1.1, 1.2]:
                        # print(deg)
                        # deg_rad = math.radians(deg)
                        M = cv.getRotationMatrix2D((thresh.shape[1] // 2, thresh.shape[0] // 2), deg, scale)
                        thresh_rot = cv.warpAffine(thresh, M, (thresh.shape[1], thresh.shape[0]))
                        if erosion > 0:
                            thresh_rot = cv.erode(thresh_rot, np.ones((erosion, erosion), np.uint8), iterations=1)
                        elif erosion < 0:
                            thresh_rot = cv.dilate(thresh_rot, np.ones((-erosion, -erosion), np.uint8), iterations=1)
                        # show_image('img', thresh_rot)
                        templates[number].append(thresh_rot)
            # thresh = cv.erode(thresh, np.ones((2, 2), np.uint8), iterations=1)
            # thresh = cv.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
            # thresh = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
            
            # show_image('img', thresh)
            # templates[number].append(thresh)
    # show_in_grid()
            
    
    print("-*-"*10)
    print("DONE: SETTING UP FOLDERS")
    print("-*-"*10)

    

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
    for file_name in sorted(os.listdir(FOLDER_IMAGES)):
        round, turn = file_name.split("_")
        round = int(round)
        turn_int = int(turn.split(".")[0]) if (turn.split(".")[0]).isdigit() else 0
        # if turn_int > 3:
        #     continue
        # if fast and turn.startswith('10'):
        #     break
        # print(round, turn)
        file_path = os.path.join(FOLDER_IMAGES, file_name)
        if round not in rez:
            rez[round] = {
                "paths": [],
                "turns": ''
            }
        if turn.endswith("turns.txt"):
            file_content = open(file_path, "r").readlines()
            rez[round]["turns"] = [line.strip().split(' ') for line in file_content]
            rez[round]["turns"] = [(player, int(turn)) for player, turn in rez[round]["turns"]]
        elif turn.endswith(".jpg"):
            # img = cv.imread(file_path)
            rez[round]["paths"].append(file_path)
    print("-*-"*10)
    print("DONE: FILES READ")
    print("-*-"*10)
    return rez


def preditct_number(game_board, current_points, patch, patch_padded, y, x, deep_pieces, verbose=0):
    possible_results = set()

    # get the possible numbers
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if abs(i) == abs(j):
                continue

            nb_1_x = x + i
            nb_1_y = y + j
            if nb_1_x < 0 or nb_1_x >= boxes or nb_1_y < 0 or nb_1_y >= boxes:
                continue
            if (nb_1_y, nb_1_x) not in current_points:
                continue
            if game_board[nb_1_y][nb_1_x] == '' or (game_board[nb_1_y][nb_1_x] in initial_signs and not game_board[nb_1_y][nb_1_x].isnumeric()):
                continue
            
            nb_2_x = nb_1_x + i
            nb_2_y = nb_1_y + j
            if nb_2_x < 0 or nb_2_x >= boxes or nb_2_y < 0 or nb_2_y >= boxes:
                continue
            if (nb_2_y, nb_2_x) not in current_points:
                continue
            if game_board[nb_2_y][nb_2_x] == '' or (game_board[nb_2_y][nb_2_x] in initial_signs and not game_board[nb_2_y][nb_2_x].isnumeric()):
                continue

            nb_1 = game_board[nb_1_y][nb_1_x]
            nb_2 = game_board[nb_2_y][nb_2_x]

            if isinstance(nb_1, str) and not nb_1.isnumeric():
                continue
            if isinstance(nb_2, str) and not nb_2.isnumeric():
                continue

            nb_1 = isinstance(nb_1, str) and int(nb_1) or nb_1
            nb_2 = isinstance(nb_2, str) and int(nb_2) or nb_2

            isPlus = game_board[y][x] == '+'
            isMinus = game_board[y][x] == '-'
            isTimes = game_board[y][x] == 'x'
            isDivide = game_board[y][x] == '/'

            isSafe = not (isPlus or isMinus or isTimes or isDivide)

            if (nb_1 + nb_2) in numbers and (isSafe or isPlus): 
                possible_results.add(nb_1 + nb_2)
            if nb_1 - nb_2 in numbers and nb_1 - nb_2 >= 0 and (isSafe or isMinus):
                possible_results.add(nb_1 - nb_2)
            if nb_2 - nb_1 in numbers and nb_2 - nb_1 >= 0 and (isSafe or isMinus):
                possible_results.add(nb_2 - nb_1)
            if (nb_1 * nb_2) in numbers and (isSafe or isTimes):
                possible_results.add(nb_1 * nb_2)
            if nb_2 != 0 and nb_1 // nb_2 in numbers and nb_1 % nb_2 == 0 and (isSafe or isDivide):
                possible_results.add(nb_1 // nb_2)
            if nb_1 != 0 and nb_2 // nb_1 in numbers and nb_2 % nb_1 == 0 and (isSafe or isDivide):
                possible_results.add(nb_2 // nb_1)

            if verbose >= 2:
                print("Possible results", possible_results)
            # print("Possible results", possible_results)
            # print(possible_results)
            if verbose >= 2:
                print('nb_1', nb_1, 'nb_2', nb_2)
                print('nb_1_x', nb_1_x, 'nb_1_y', nb_1_y)
                print('nb_2_x', nb_2_x, 'nb_2_y', nb_2_y)
                print('x', x, 'y', y)
                show_image('patch', patch_padded)
    final_possible_results = []
    for nb in possible_results:
        candidate_list = templates[nb]
        img_gray = cv.cvtColor(patch_padded, cv.COLOR_BGR2GRAY)
        img_gray = sharp_n_thresh(img_gray)
        #add a margin of 30 pixels
        img_gray = cv.copyMakeBorder(img_gray, 30, 30, 30, 30, cv.BORDER_CONSTANT, value=[0, 0, 0])
        if verbose >= 2:
            show_image('img_gray', img_gray)
        pos_scores = []
        for candidate in candidate_list:
            res = cv.matchTemplate(img_gray,candidate,cv.TM_CCORR_NORMED)
            score = np.max(res)
            # if verbose >= 2:
            #     print('score', nb, score)
            #     show_image('res', res)
            pos_scores.append(score)
        score = np.max(pos_scores)
        # print('alost_final', score, nb)
        if deep_pieces[nb] <= 0:
            score /= 2
        final_possible_results.append((score, nb))
    final_possible_results = sorted(final_possible_results, key=lambda x: x[0], reverse=True)
    if verbose >= 1:
        print('final_possible_results', final_possible_results)
    if len(final_possible_results) == 0:
        return 0
    return final_possible_results[0][1]


            
def get_bonus_score(game_board, current_points, y, x, current):
    current = int(current) if isinstance(current, str) else current
    bonus_sum = 0
    # print('-'*10)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if abs(i) == abs(j):
                continue
            new_x = x + i
            new_y = y + j
            if new_x < 0 or new_x >= boxes or new_y < 0 or new_y >= boxes:
                continue
            if (new_y, new_x) not in current_points:
                continue
            if game_board[new_y][new_x] == '' or (game_board[new_y][new_x] in initial_signs and not game_board[new_y][new_x].isnumeric()):
                continue
            
            new_x_2 = new_x + i
            new_y_2 = new_y + j
            if new_x_2 < 0 or new_x_2 >= boxes or new_y_2 < 0 or new_y_2 >= boxes:
                continue
            if (new_y_2, new_x_2) not in current_points:
                continue
            if game_board[new_y_2][new_x_2] == '' or (game_board[new_y_2][new_x_2] in initial_signs and not game_board[new_y_2][new_x_2].isnumeric()):
                continue
            # print('new_y', new_y)
            # print('new_x', new_x)
            nb_1 = int(game_board[new_y][new_x]) if isinstance(game_board[new_y][new_x], str) else game_board[new_y][new_x]
            nb_2 = int(game_board[new_y_2][new_x_2]) if isinstance(game_board[new_y_2][new_x_2], str) else game_board[new_y_2][new_x_2]

            # print('verificam pentru BONUS:', nb_1, nb_2, current)            
            isPlus = game_board[y][x] == '+'
            isMinus = game_board[y][x] == '-'
            isTimes = game_board[y][x] == 'x'
            isDivide = game_board[y][x] == '/'
            isSafe = not (isPlus or isMinus or isTimes or isDivide)
            
            if nb_1 + nb_2 == current and (isSafe or isPlus):
                bonus_sum += current
                continue
            if nb_1 - nb_2 == current and (isSafe or isMinus):
                bonus_sum += current
                continue
            if nb_2 - nb_1 == current and (isSafe or isMinus):
                bonus_sum += current
                continue
            if nb_1 * nb_2 == current and (isSafe or isTimes):
                bonus_sum += current
                continue
            if (not nb_2 == 0) and nb_1 // nb_2 == current and nb_1 % nb_2 == 0 and (isSafe or isDivide):
                bonus_sum += current
                continue
            if (not nb_1 == 0) and nb_2 // nb_1 == current and nb_2 % nb_1 == 0 and (isSafe or isDivide):
                bonus_sum += current
                continue
    # if bonus_sum > 0:
    #     print('ini', bonus_sum)
    bonus_sum -= current
    if bonus_sum > 0:
        # print('post bonus', bonus_sum)
        return bonus_sum
    else:
        return 0


            
    


lower_patch = np.array([0, 162, 180])
upper_patch = np.array([214, 255, 231])

# lower_patch = np.array([0, 0, 152])
# upper_patch = np.array([255, 255, 255])

init_game()
files = get_files(True)
for round in range(ROUNDS_FIRST, ROUNDS_LAST + 1):
    # print(files[str(i)])
    prev_patches = None
    prev_board = None
    current_board = deepcopy(initial_game_board)
    current_points = deepcopy(starting_points)
    turn = 1
    scores = []
    score = 0

    score_pointer = 1
    change_turn_pos = files[round]["turns"][1][1]

    deep_pieces = deepcopy(pieces)
    # for current_position in current_points:
    #     if current_position in starting_points:
    #         continue
    #     nb = game_board[current_position[0]][current_position[1]]
    #     nb = isinstance(nb, str) and int(nb) or nb
        
    
    for img_path in files[round]["paths"]:
        print(round, turn)
        try_save_from_restanta = None
        try:
            if turn == change_turn_pos:
                scores.append(score)
                print('score:', score)
                score = 0
                score_pointer += 1
                try:
                    change_turn_pos = files[round]["turns"][score_pointer][1]
                except:
                    change_turn_pos = 51
                    # score = scores[-1]


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
            patches_paddded = get_patches(res, lines_v, lines_h, padding=10)

            # rez = np.zeros((total_size, total_size, 3), dtype=np.uint8)
            # for i in range(boxes):
            #     for j in range(boxes):
            #         x = i * box_size
            #         y = j * box_size
            #         # show_image('img', patches[i][j])
            #         rez[y:y+box_size, x:x+box_size] = patches[j][i]
            
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
                    try_save_from_restanta = (chosen_y, chosen_x)
                    predicted = preditct_number(current_board, current_points, chosen, patches_paddded[chosen_y][chosen_x], chosen_y, chosen_x, deep_pieces=deep_pieces)
                    deep_pieces[predicted] -= 1
                    score += predicted
                    current_board[chosen_y][chosen_x] = predicted
                    current_points.append((chosen_y, chosen_x))
                    write_annotation_file(round, turn, format_chess_like_position(chosen_y, chosen_x), predicted)

                prev_patches = deepcopy(patches)
                prev_board = deepcopy(current_board)
            else:
                possible_points = get_possible_points(res, current_board, current_points)
                # print(round, img_path, turn)
                # for current_position in current_points:
                #     y, x = current_position
                #     cv.circle(rez, (x * box_size + box_size // 2, y * box_size + box_size // 2), 10, (0, 0, 255), -1)
                # for possible_point in possible_points:
                #     y, x = possible_point
                #     cv.circle(rez, (x * box_size + box_size // 2, y * box_size + box_size // 2), 10, (0, 255, 0), -1)
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
                    # print('turn:', turn)
                    # verb = 2 if turn == 41 else 0
                    verb = 0
                    if verb >= 2:
                        show_image('res', res)
                    predicted = preditct_number(current_board, current_points, chosen, patches_paddded[chosen_y][chosen_x], chosen_y, chosen_x, deep_pieces=deep_pieces, verbose=verb)
                    deep_pieces[predicted] -= 1
                    bonus = get_bonus_score(current_board, current_points, chosen_y, chosen_x, predicted)
                    # print('BONUS:', bonus)
                    if current_board[chosen_y][chosen_x] == '3x':
                        # print('triplu')
                        score += 3 * (predicted + bonus)
                    elif current_board[chosen_y][chosen_x] == '2x':
                        # print('dublu')
                        score += 2 * (predicted + bonus)
                    else:
                        # print('simplu')
                        score += predicted + bonus
                    # print('score_now:', score)
                    current_board[chosen_y][chosen_x] = predicted
                    current_points.append((chosen_y, chosen_x))
                    # show_image('img', chosen_patch)
                    # show_where = res.copy()
                    # cv.circle(show_where, (chosen_x * box_size + box_size // 2, chosen_y * box_size + box_size // 2), 10, (0, 0, 255), -1)
                    # show_image('where', show_where)
                    write_annotation_file(round, turn, format_chess_like_position(chosen_y, chosen_x), predicted)

                prev_board = current_board.copy()
                prev_patches = patches

            # show_image('img', rez)
            turn += 1
        except Exception as e:
            print('Error:', round, img_path, turn)
            print(e)
            if try_save_from_restanta is not None:
                current_board[try_save_from_restanta[0]][try_save_from_restanta[1]] = 1
                write_annotation_file(round, turn, format_chess_like_position(try_save_from_restanta[0], try_save_from_restanta[1]), 1)
            continue
        
        rez_score = ''
        idx = 0
        # print(scores)
        for lines in files[round]["turns"]:
            # print(lines, idx)
            if idx >= len(scores):
                rez_score += f"{lines[0]} {lines[1]} {score}"
                break
            rez_score += f"{lines[0]} {lines[1]} {scores[idx]}\n"
            idx += 1
        write_score_file(round, rez_score)
# print(files)

