#! /usr/bin/env python
# tree.py
#
# Written by Doug Dahms
# modified by glallen @ StackExchange
#
# Prints the tree structure for the path specified on the command line

from os import listdir, sep
from os.path import abspath, basename, isdir
from sys import argv

def tree(dir, padding, print_files=False, limit=10000):
    print( padding[:-1] + '+-' + basename(abspath(dir)) + '/' )
    padding = padding + ' '
    limit = int(limit)
    files = []
    if print_files:
        files = listdir(dir)
    else:
        files = [x for x in listdir(dir) if isdir(dir + sep + x)]
    count = 0
    for file in files:
        count += 1
        path = dir + sep + file
        if isdir(path):
            print( padding + '|' )
            if count == len(files):
                tree(path, padding + ' ', print_files, limit)
            else:
                tree(path, padding + '|', print_files, limit)
        else:
            if limit == 10000:
                print( padding + '|' )
                print( padding + '+-' + file )
                continue
            elif limit == 0:
                print( padding + '|' )
                print( padding + '+-' + '... <additional files>' )
                limit -= 1
            elif limit <= 0:
                continue
            else:
                print( padding + '|' )
                print( padding + '+-' + file )
                limit -= 1

def usage():
    return '''Usage: %s [-f] [file-listing-limit(int)] <PATH>
Print tree structure of path specified.
Options:
-f          Print files as well as directories
-f [limit]  Print files as well as directories up to number limit
PATH        Path to process''' % basename(argv[0])

def main():
    if len(argv) == 1:
        print( usage() )
    elif len(argv) == 2:
        # print just directories
        path = argv[1]
        if isdir(path):
            tree(path, ' ')
        else:
            print( 'ERROR: \'' + path + '\' is not a directory' )
    elif len(argv) == 3 and argv[1] == '-f':
        # print directories and files
        path = argv[2]
        if isdir(path):
            tree(path, ' ', True)
        else:
            print( 'ERROR: \'' + path + '\' is not a directory' )
    elif len(argv) == 4 and argv[1] == '-f':
        # print directories and files up to max
        path = argv[3]
        if isdir(path):
            tree(path, ' ', True, argv[2])
        else:
            print( 'ERROR: \'' + path + '\' is not a directory' )
    else:
        print( usage() )

if __name__ == '__main__':
    main()