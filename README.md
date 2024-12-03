# Requirements:
numpy==2.1.3
opencv_python==4.10.0.84

## Added requirements for the project in the requirements.txt file
python install -r requirements.txt

# Usage:
- All the tasks run from the score.py file, in the main function.
  - To run all the tasks, run the score.py file
- The configuration for the tasks is done in the config.py file
    - FOLDER_IMAGES for the folder where the images are stored(fake_test)
    - FOLDER_SCORES for the folder where the scores are stored(./fisiere_solutie/342_Chirus_Mina_Sebastian/)
    - FOLDER_GROUND_TRUTH for the folder where the ground truth is stored(fake_test_gt)
- To run the tests, run the game.py file. It uses the same configuration as the score.py file