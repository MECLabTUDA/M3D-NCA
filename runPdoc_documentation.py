import pdoc
import os

"""https://pdoc.dev/docs/pdoc.html"""

def addFileToDocumentation(path):
    os.system('pdoc --html ' + path + ' --force')

def main():
    os.system('pdoc src ') #--force --html

if __name__ == '__main__':
    main()

