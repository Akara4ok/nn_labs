import argparse
import pathlib
import sys
sys.path.append('config')
import settings

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-path", "-d", default=settings.DATA_PATH, help="path to dataset", type=str)
    parser.add_argument("--save-path", "-s", default=settings.LABELS_PATH, help="labels path to save", type=str)
    args = vars(parser.parse_args())
    
    data_dir = pathlib.Path(args['data_path'])
    class_names = sorted([item.name for item in data_dir.glob('*')])
    file_str = '\n'.join(class_names)
    with open(args['save_path'], "w") as file:
        file.write(file_str)