#####################################
# TRAIN
#####################################
# FOLDER_IMAGES = "./antrenare"
# FOLDER_SCORES = "./fisiere_solutie/342_Chirus_Mina_Sebastian/"
# FOLDER_GROUND_TRUTH = "./antrenare/"
# ROUNDS_FIRST = 1
# ROUNDS_LAST = 4 # range(ROUNDS_START, ROUNDS_END + 1) '+1' is included when running

#####################################
# TEST
#####################################
# FOLDER_IMAGES = "./fake_test"
# FOLDER_SCORES = "./fisiere_solutie/fake_test/342_Chirus_Mina_Sebastian/"
# FOLDER_GROUND_TRUTH = "./fake_test_gt/"
# ROUNDS_FIRST = 1
# ROUNDS_LAST = 1 # range(ROUNDS_START, ROUNDS_END + 1) '+1' is included when running


#####################################
# REAL GAME
#####################################
FOLDER_IMAGES = "../testare/"
FOLDER_SCORES = "./fisiere_solutie/342_Chirus_Mina_Sebastian/"
FOLDER_GROUND_TRUTH = "./cod_evaluare/evaluare_gt/"
ROUNDS_FIRST = 1
ROUNDS_LAST = 4 # range(ROUNDS_START, ROUNDS_END + 1) '+1' is included when running




#####################################
# DO NOT CHANGE
#####################################
FOLDER_TEMPLATES = "./templates/patches/"
box_size = 90
boxes = 14
total_size = box_size * boxes
something_wrong_threshold = 20
search_range = 10
game_board = [
    ['3x', '', '', '', '', '', '3x', '3x', '', '', '', '', '', '3x'],
    ['', '2x', '', '', '/', '', '', '', '', '/', '', '', '2x', ''],
    ['', '', '2x', '', '', '-', '', '', '-', '', '', '2x', '', ''],
    ['', '', '', '2x', '', '', '+', 'x', '', '', '2x', '', '', ''],
    ['', '/', '', '', '2x', '', 'x', '+', '', '2x', '', '', '/', ''],
    ['', '', '-', '', '', '', '', '', '', '', '', '-', '', ''],
    ['3x', '', '', 'x', '+', '', '1', '2', '', 'x', '+', '', '', '3x'],
    ['3x', '', '', '+', 'x', '', '3', '4', '', '+', 'x', '', '', '3x'],
    ['', '', '-', '', '', '', '', '', '', '', '', '-', '', ''],
    ['', '/', '', '', '2x', '', '+', 'x', '', '2x', '', '', '/', ''],
    ['', '', '', '2x', '', '', 'x', '+', '', '', '2x', '', '', ''],
    ['', '', '2x', '', '', '-', '', '', '-', '', '', '2x', '', ''],
    ['', '2x', '', '', '/', '', '', '', '', '/', '', '', '2x', ''],
    ['3x', '', '', '', '', '', '3x', '3x', '', '', '', '', '', '3x'],
]
initial_signs = [sign for row in game_board for sign in row if sign != '']
starting_points = [(6, 6), (6, 7), (7, 6), (7, 7)]
pieces = {
    0: 1,
    1: 7,
    2: 7,
    3: 7,
    4: 7,
    5: 7,
    6: 7,
    7: 7,
    8: 7,
    9: 7,
    10: 7,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1,
    21: 1,
    24: 1,
    25: 1,
    27: 1,
    28: 1,
    30: 1,
    32: 1,
    35: 1,
    36: 1,
    40: 1, 
    42: 1,
    45: 1,
    48: 1,
    49: 1,
    50: 1,
    54: 1,
    56: 1,
    60: 1,
    63: 1,
    64: 1,
    70: 1,
    72: 1,
    80: 1,
    81: 1,
    90: 1
}
numbers = [k for k in pieces.keys()]