import os
import json

key = 1
amount = 0

def inputfile():
    path = "DialogSum_Data/"
    files_list=os.listdir(path)
        
    key_word=str(input("Please enter the name of original file: "))     

    for file_name in files_list:
        if key_word in file_name:
            original_file = path + file_name         
    
    return original_file
            

def copydata(original_file):
    global amount
    samsize = int(input("How many lines you'd like to take from the file? -- ")) 
    amount = amount + samsize
    with open(original_file, 'r', encoding = 'utf-8') as f1:
        with open('DialogSum_Data/data.autosample.jsonl','a') as fs:
            n= -1
            for line in f1:
                n=n+1
                if n < samsize:
                    fs.write(line)                
             

def again():
    global key
    check = input("Would you like to add some more sample to the pool? (y/n) -- ")
    if check != 'y':
        key = 0 
        
    
def main():
    print("Let's create a sample data pool.")
    while key == 1:
        my_input = inputfile()
        copydata(my_input)
        again() 
    print(''' --------Job Completed--------
The sample pool is created with {} data.
Thank you for using the sample creation tool. 
--------Enjoy your coding!--------'''.format(amount))


main()