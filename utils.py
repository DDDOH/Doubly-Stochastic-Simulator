import shutil
import datetime
import os

def create_result_folder(result_dir, exp_label):
    # copy current folder to a result directory
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join(result_dir, exp_label, time_str)
    # copy all the files in cwd to result_dir, except for the 'data.csv' file and the 'results' folder
    shutil.copytree('.', os.path.join(result_dir, 'source'), ignore=shutil.ignore_patterns('*.csv', '.git', 'results'))

    print('Copy code to ' + result_dir)

    # save experiment information to the result folder
    import sys
    import time
    txt_file_name = 'Experiment information.txt'

    str_cmd = 'terminal command for this result folder:\npython '
    for _ in sys.argv:
        str_cmd += _
    str_cmd += '\n\n'

    text_file = open(os.path.join(result_dir, txt_file_name), "w")
    n = text_file.write(str_cmd)

    # get current time and time zone    
    str_time = 'Job start time: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ', time zone: ' + str(time.tzname[1])
    str_time += '\n\n'
    text_file.write(str_time)

    text_file.write('Current working directory: ' + os.getcwd())
    text_file.close()

    return result_dir