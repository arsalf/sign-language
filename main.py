from numpy import sign
from signlanguage import SignLanguage
import os
import h5py
import glob

DATA_PATH = os.path.join('MP_Data')
VIDEOS = 20
FRAME = 30
MODEL_FILENAME = 'model_sign_language.h5'

def main_menu():
    print("========= AUDIRE =========")
    print("1. Trainer")
    print("2. Test")
    print("3. Data Mining")
    print("0. Exit")
    print("==========================")
    print("Choose : ")

def clear():
    os.system("cls")

def get_actions():
    actions = []
    for action in glob.glob(DATA_PATH+'/*'):                
        actions.append(action.split('\\')[1])
    return actions

def main():

    main_menu()
    choose = int(input())

    clear()
    
    if choose == 1:
        actions = get_actions()
        sign = SignLanguage(DATA_PATH, actions, VIDEOS, FRAME)
        sign.realtime_trainer()
    elif choose == 2:
        actions = get_actions()
        sign = SignLanguage(DATA_PATH, actions, VIDEOS, FRAME)    
        sign.realtime_test(MODEL_FILENAME, 0)
    elif choose == 3:
        print("Berapa kata/huruf yang akan dimasukan ? ")
        no_actions = int(input())
        print("=======================")                 

        #input actions sign language 
        actions=[]
        for action in range(no_actions):           
            print("Kata/Huruf ke-"+str(action+1)+" :")
            actions.append(input())
        clear()
        
        print("Oke, siap untuk memasukan data secara real-time ya!")
        sign = SignLanguage(DATA_PATH, actions, VIDEOS, FRAME)
        sign.make_data_model(0)
    else:
        exit()

if __name__ == '__main__':
    main()