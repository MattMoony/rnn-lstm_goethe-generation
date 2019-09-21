import os
from tqdm import tqdm
from argparse import ArgumentParser

def concat_files(p):
    total = ''
    for fpath in tqdm(os.listdir(p), desc='Processing files', mininterval=1e-9):
        with open(os.path.join(p, fpath), 'rb') as f:
            total += ''.join([s.decode('utf-8') for s in f.readlines() 
                                if s.decode('utf-8') != ''])
    return total

def save_total(content, p):
    with open(p, 'wb') as f:
        f.write(content.encode('utf-8'))

def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', type=str, dest='src', 
                        help='Source directory', default='data/')
    parser.add_argument('-d', '--dest', type=str, dest='dest', 
                        help='Destination file', default='data.txt')

    args = parser.parse_args()

    all_content = concat_files(args.src)
    save_total(all_content, args.dest)

if __name__ == '__main__':
    main()