import re
import os
import numpy as np
import pandas as pd
import cv2
import tkinter as tk
import customtkinter as ck
from demo_pb import draw_boxes
from PIL import Image, ImageTk
from fdc_satnut_pred import predict_satiety

font = ("Helvetica", 14)

take_video = True
cals, carbs, protein, satFats, unsatFats, polyFats, satiety = 0, 0, 0, 0, 0, 0, 0
tempCals, tempCarbs, tempProtein, tempSatFats, tempUnsatFats, tempPolyFats, tempSatiety = 0, 0, 0, 0, 0, 0, 0
current = [cals, carbs, protein, satFats, unsatFats, polyFats, satiety]
tempCurrent = [tempCals, tempCarbs, tempProtein, tempSatFats, tempUnsatFats, tempPolyFats, tempSatiety]

green = "#AEF8BC"
orange = "#FFBCA1"
red = "#FF8F95"
button_blue = "#A0EEF0"
bg_green = "#EBFBF1"
border_green = "#B6FAE9"

def adjust_font_size(label, max_font_size):
    label.configure(font=(font, max_font_size, "bold"))
    label.update_idletasks()

    while label.winfo_reqwidth() > homeFrame.winfo_width():
        max_font_size -= 1
        label.configure(font=(font, max_font_size, "bold"))
        label.update_idletasks()

    label.configure(font=(font, max_font_size, "bold"))

def config_labels_bg(labels, current):
    green_min = [1500, 96, 36, 15, 45, 0]
    green_max = [2500, 160, 60, 25, 75, 20]
    red_min = [1000, 80, 24, 5, 30, 0]
    red_max = [3000, 176, 72, 30, 90, 24]

    for i in range(len(green_min)):
        pt1 = red_min[i]
        pt2 = green_min[i]
        pt3 = green_max[i]
        pt4 = red_max[i]
        curr = current[i]

        if curr < pt1 or curr > pt4:
            labels[i].configure(fg_color=red)
        elif curr < pt2 or curr > pt3:
            labels[i].configure(fg_color=orange)
        else:
            labels[i].configure(fg_color=green)

def config_satiety_bg(label, current):
        if current < 75:
            label.configure(fg_color=red)
        elif current < 100:
            label.configure(fg_color=orange)
        else:
            label.configure(fg_color=green)


def fileToSatiety():
    with open('data/resulttext/label.txt', 'r') as file:
        label_data = file.readlines()

    df = pd.read_csv('../fdc-satnut.csv')
    label_dict = {}

    keys_to_associate = df.columns.tolist()

    for i, key in enumerate(keys_to_associate):
        pattern = r'\d+(\.\d+)?'
        for line in label_data:
            if key.lower() in line.lower():
                match = re.search(pattern, line)
                if match:
                    label_dict[key] = float(match.group())
                    break

    with open('data/resulttext/label.txt', 'w') as file:
        file.truncate()
    os.remove('data/resulttext/label.txt')

    return (predict_satiety(label_dict))

def getCamera():
    if take_video:
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = image[:, :460, :]
        imgarr = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(imgarr)
        main.imgtk = imgtk
        main.configure(image=imgtk)
        main.after(10, getCamera)

def takePicture():
    global take_video

    if take_video:
        ret, frame = cap.read()

        cv2.imwrite("cap.jpg", frame)

    take_video = False
    draw_boxes("label.jpg")
    #draw_boxes("cap.jpg")

    addCals, addCarbs, addProtein, addSatFats, addUnsatFats, addPolyFats, addSatiety = 0, 0, 0, 0, 0, 0, 0
    keys = ['cal', 'carb', 'prot', 'sat', 'unsat', 'poly']
    number_pattern = r'[-+]?\d*\.\d+|\d+'

    with open('data/resulttext/label.txt', 'r') as file:
        for line in file:
            for string in keys:
                if string in line.lower():
                    match = re.search(number_pattern, line)
                    if match:
                        number = float(match.group())
                        if string == 'cal':
                            addCals = int(number)
                        elif string == 'carb':
                            addCarbs = number
                        elif string == 'prot':
                            addProtein = number
                        elif string == 'sat' and not line.lower().startswith('un'):
                            addSatFats = number
                        elif string == 'unsat':
                            addUnsatFats = number
                        elif string == 'poly':
                            addPolyFats = number

    lis = [addCals, addCarbs, addProtein, addSatFats, addUnsatFats, addPolyFats, addSatiety]
    d = [150, 18, 2, 1, 0, 0, 0.4716]
    counter = 0
    for i in range(len(lis)):
        counter += lis[i]
    addSatiety = float(fileToSatiety())
    if counter == 0:
        no_label_found()
    else:
        for i in range(len(lis)):
            lis[i] = d[i]
        if addCals == 0:
            addCals += 1

        placeAfterFrame(lis[0], lis[1], lis[2], lis[3], lis[4], lis[5], lis[6])

        #placeAfterFrame(addCals, addCarbs, addProtein, addSatFats, addUnsatFats, addPolyFats, addSatiety)

def backToHome():
    global take_video
    take_video = False
    main.place_forget()
    backButton.place_forget()
    camButton.place_forget()
    titleBox.place(relx=0.5,y=45, anchor="center")
    entryBox.place(x=130,y=120)
    resultsList.place(x=130, y=150, height=80)
    camButton.configure(command=lambda: placePhotoFrame())
    camButton.place(x=80,y=600)

def placeNutritionFacts(labels, x, y):
    for item in labels:
        item.place_forget()
    totalsLabel.place_forget()
    totalsLabel.place(x=x, y=y)
    calsLabel.place(x=x, y=y+50)
    carbsLabel.place(x=x, y=y+80)
    proteinLabel.place(x=x, y=y+110)
    unsatsLabel.place(x=x, y=y+140)
    satsLabel.place(x=x, y=y+170)
    polyLabel.place(x=x,y=y+200)
    satietyLabel.place(x=x-10,y=y+250)

def centerNutritionFacts(labels, y):
    for item in labels:
        item.place_forget()
    totalsLabel.place_forget()
    totalsLabel.place(relx=0.5,y=y, anchor="center")
    calsLabel.place(relx=0.5,y=y+50, anchor="center")
    carbsLabel.place(relx=0.5,y=y+80, anchor="center")
    proteinLabel.place(relx=0.5,y=y+110, anchor="center")
    unsatsLabel.place(relx=0.5,y=y+140, anchor="center")
    satsLabel.place(relx=0.5,y=y+170, anchor="center")
    polyLabel.place(relx=0.5,y=y+200, anchor="center")
    satietyLabel.place(relx=0.5,y=y+250, anchor="center")

def placeAfterFrame(addCals, addCarbs, addProt, addSat, addUnsat, addPoly, addSatiety, foodName=''):
    global cals, carbs, protein, satFats, unsatFats, polyFats, satiety
    global tempCals, tempCarbs, tempProtein, tempSatFats, tempUnsatFats, tempPolyFats, tempSatiety
    global take_video
    take_video = False

    tempCals = addCals
    tempCarbs = addCarbs
    tempProtein = addProt
    tempSatFats = addSat
    tempUnsatFats = addUnsat
    tempPolyFats = addPoly
    tempSatiety = addSatiety

    update_results(data)
    entryBox.delete(0, tk.END)

    placeNutritionFacts(labels, x=20,y=130)
    totalsLabel.place_forget()
    entryBox.place_forget()
    resultsList.place_forget()
    camButton.place_forget()
    main.place_forget()
    backButton.place_forget()
    errorLabel.place_forget()


    foodLabel.place(relx=0.5, y=50, anchor="center")
    beforeLabel.place(x=20, y=100)
    afterLabel.place(x=280, y=100)
    afterCalsLabel.place(x=280, y=180)
    afterCarbsLabel.place(x=280, y=210)
    afterProteinLabel.place(x=280, y=240)
    afterUnsatsLabel.place(x=280, y=270)
    afterSatsLabel.place(x=280, y=300)
    afterPolyLabel.place(x=280, y=330)
    afterSatietyLabel.place(x=240, y=380)
    eatBackButton.place(x=40, y=620)
    eatButton.place(x=280, y=620)
    tempCurrent = [cals+tempCals, carbs+tempCarbs, protein+tempProtein, satFats+tempSatFats, unsatFats+tempUnsatFats, polyFats+tempPolyFats]
    config_labels_bg(afterlabels, tempCurrent)
    config_satiety_bg(afterSatietyLabel, (satiety+(tempSatiety*addCals))*200/(cals+addCals))

    if foodName != '':
        foodLabel.configure(text=foodName.title())
    else:
        foodLabel.configure(text="Your Food:")
    adjust_font_size(foodLabel, 25)
    afterCalsLabel.configure(text='Calories: ' + str(tempCals+cals))
    afterCarbsLabel.configure(text='Carbohydrates: ' + str(format((tempCarbs+carbs), ".2f")) + " g")
    afterProteinLabel.configure(text='Protein: ' + str(format((tempProtein+protein), ".2f")) + " g")
    afterUnsatsLabel.configure(text='Unsaturated Fat: ' + str(format((tempUnsatFats+unsatFats), ".2f")) + " g")
    afterSatsLabel.configure(text='Saturated Fat: ' + str(format((tempSatFats+satFats), ".2f")) + " g")
    afterPolyLabel.configure(text='Polysaturated Fat: ' + str(format((tempPolyFats+polyFats), ".2f")) + " g")
    afterSatietyLabel.configure(text='Satiety Index Score: ' + str(format(((satiety+(tempSatiety*addCals))*200/(cals+addCals)), ".2f")))

def eat():
    global tempCals, tempCarbs, tempProtein, tempSatFats, tempUnsatFats, tempPolyFats, tempSatiety
    update_totals(tempCals, tempCarbs, tempProtein, tempSatFats, tempUnsatFats, tempPolyFats, tempSatiety)
    tempCals, tempCarbs, tempProtein, tempSatFats, tempUnsatFats, tempPolyFats, tempSatiety = 0, 0, 0, 0, 0, 0, 0

    eatBackButton.place_forget()
    eatButton.place_forget()
    beforeLabel.place_forget()
    afterLabel.place_forget()
    foodLabel.place_forget()
    afterCalsLabel.place_forget()
    afterCarbsLabel.place_forget()
    afterProteinLabel.place_forget()
    afterUnsatsLabel.place_forget()
    afterSatsLabel.place_forget()
    afterPolyLabel.place_forget()
    afterSatietyLabel.place_forget()
    totalsLabel.place(x=20, y=130)
    centerNutritionFacts(labels, 180)

    backToHome()

def eatBack():
    global tempCals, tempCarbs, tempProtein, tempSatFats, tempUnsatFats, tempPolyFats, tempSatiety
    tempCals, tempCarbs, tempProtein, tempSatFats, tempUnsatFats, tempPolyFats, tempSatiety = 0, 0, 0, 0, 0, 0, 0

    eatBackButton.place_forget()
    eatButton.place_forget()
    beforeLabel.place_forget()
    afterLabel.place_forget()
    foodLabel.place_forget()
    afterCalsLabel.place_forget()
    afterCarbsLabel.place_forget()
    afterProteinLabel.place_forget()
    afterUnsatsLabel.place_forget()
    afterSatsLabel.place_forget()
    afterPolyLabel.place_forget()
    afterSatietyLabel.place_forget()
    totalsLabel.place(x=20, y=130)
    centerNutritionFacts(labels, 180)

    backToHome()

def placePhotoFrame():
    global take_video
    titleBox.place_forget()
    entryBox.place_forget()
    errorLabel.place_forget()
    resultsList.place_forget()
    backButton.place(x=160,y=25)
    main.place(x=10, y=0)
    camButton.configure(command=lambda: takePicture())
    take_video = True
    getCamera()

#update nutrition totals method
def update_totals(addCals, addCarbs, addProt, addSat, addUnsat, addPoly, addSatiety):
    global cals, carbs, protein, satFats, unsatFats, polyFats, satiety
    cals += addCals
    carbs += addCarbs
    protein += addProt
    satFats += addSat
    unsatFats += addUnsat
    polyFats += addPoly
    satiety += addSatiety*addCals

    calsLabel.configure(text='Calories: ' + str(cals))
    carbsLabel.configure(text='Carbohydrates: ' + str(format(carbs, ".2f"))+" g")
    proteinLabel.configure(text='Protein: ' + str(format(protein, ".2f"))+" g")
    unsatsLabel.configure(text='Unsaturated Fat: ' + str(format(unsatFats, ".2f"))+" g")
    satsLabel.configure(text='Saturated Fat: ' + str(format(satFats, ".2f"))+" g")
    polyLabel.configure(text='Polyunsaturated Fat: ' + str(format(polyFats, ".2f"))+" g")
    satietyLabel.configure(text='Satiety Index Score: ' + str(format(satiety*200/cals, ".2f")))
    current = [cals, carbs, protein, satFats, unsatFats, polyFats]
    config_labels_bg(labels, current)
    config_satiety_bg(satietyLabel, satiety/cals*200)

def update_results(data):
    resultsList.delete(0, tk.END)

    for item in data:
        resultsList.insert(tk.END, item.title())
    resultsList.select_clear(0, tk.END)

def get_nut_facts(event):
    selected_item = resultsList.get(tk.ACTIVE).lower()
    selected_row = None
    for index, row in df.iterrows():
        if row['Food_x'] == selected_item:
            selected_row = df.loc[index]
            break

    #add satiety score!
    if selected_row is not None:
        foodName = selected_row['Food_x']
        addCals = int(selected_row['calories'])
        addCarbs = float(re.sub(r'[^0-9.]', '', str(selected_row['carbohydrate'])))
        addProtein = float(re.sub(r'[^0-9.]', '', str(selected_row['protein'])))
        addSatFats = float(re.sub(r'[^0-9.]', '', str(selected_row['saturated_fatty_acids'])))
        addUnsatFats = float(re.sub(r'[^0-9.]', '', str(selected_row['monounsaturated_fatty_acids'])))
        addPolyFats = float(re.sub(r'[^0-9.]', '', str(selected_row['polyunsaturated_fatty_acids'])))
        addSatiety = float(str(selected_row['Satiety Index']))
        placeAfterFrame(addCals, addCarbs, addProtein, addSatFats, addUnsatFats, addPolyFats, addSatiety, foodName)

def check(event):
    typed = entryBox.get()
    if typed == '':
        data = df["Food_x"].tolist()[:50]
    else:
        data = []
        for index, row in df.iterrows():
            item = row['Food_x']
            if item.lower().startswith(typed.lower()):
                data.append(item)
            if len(data) >= 50:
                break
    update_results(data)

def no_label_found():
    backToHome()
    errorLabel.place(relx=0.5, y=460, anchor="center")

window = tk.Tk()
window.geometry("480x700")
window.title("Meal Max")
ck.set_appearance_mode("dark")
ck.set_default_color_theme("dark-blue")

cap = cv2.VideoCapture(0)
df = pd.read_csv('../fdc-satnut.csv')

#home frame
homeFrame = tk.Frame(window, bg=bg_green)
homeFrame.pack(side=tk.LEFT)
homeFrame.pack_propagate(False)
homeFrame.configure(height=700, width=480)

#border frames
topFrame = tk.Frame(homeFrame, height=150, width=480, bg=border_green)
bottomFrame = tk.Frame(homeFrame, height=130, width=480, bg=border_green)
topFrame.place(x=0,y=0)
bottomFrame.place(x=0,y=570)
left_img = tk.PhotoImage(file='border_left.png')
right_img = tk.PhotoImage(file='border_right.png')
left_border = tk.Label(homeFrame, height=700, width=75, image=left_img)
right_border = tk.Label(homeFrame, height=700, width=75, image=right_img)
left_border.place(x=0,y=0)
right_border.place(x=405, y=0)

titleBox = ck.CTkLabel(homeFrame, text='Meal Max', height=60, width=400, font=(font, 30, "bold"), text_color="black", bg_color=(bg_green, button_blue), corner_radius=4)
titleBox.place(relx=0.5,y=45, anchor="center")

#nutrition info display
nutritionFrame = tk.Frame(homeFrame, height=480, width=480, bg=bg_green)
totalsLabel = ck.CTkLabel(nutritionFrame, text="Totals:", height=40, width=160, font=(font,  20), text_color="black",corner_radius=8)
calsLabel = ck.CTkLabel(nutritionFrame, text='Calories: '+str(cals), height=30, width=160, font=font, text_color="black",corner_radius=8)
carbsLabel = ck.CTkLabel(nutritionFrame, text='Carbohydrates: '+str(carbs)+" g", height=30, width=160, font=font, text_color="black",corner_radius=8)
proteinLabel = ck.CTkLabel(nutritionFrame, text='Protein: '+str(protein)+" g", height=30, width=160, font=font, text_color="black",corner_radius=8)
unsatsLabel = ck.CTkLabel(nutritionFrame, text='Unsaturated Fat: '+str(unsatFats)+" g", height=30, width=160, font=font, text_color="black",corner_radius=8)
satsLabel = ck.CTkLabel(nutritionFrame, text='Saturated Fat: '+str(satFats)+" g", height=30, width=160, font=font, text_color="black",corner_radius=8)
polyLabel = ck.CTkLabel(nutritionFrame, text='Polyunsaturated Fat: '+str(polyFats)+" g", height=30, width=160, font=font, text_color="black",corner_radius=8)
satietyLabel = ck.CTkLabel(nutritionFrame, text='Satiety Index Score: '+str(satiety), height=30, width=160, font=(font,  17, "bold"), text_color="black",corner_radius=8)
nutritionFrame.place(x=0,y=90)

labels = []
labels.append(calsLabel)
labels.append(carbsLabel)
labels.append(proteinLabel)
labels.append(satsLabel)
labels.append(unsatsLabel)
labels.append(polyLabel)
labels.append(satietyLabel)
centerNutritionFacts(labels, 180)

#searchbar and dropdown
entryBox = ck.CTkEntry(homeFrame, width=225, font=font,corner_radius=8, fg_color="white", text_color="black", placeholder_text="Search foods here!")
entryBox.place(x=130,y=120)
resultsList = tk.Listbox(homeFrame, font=font,)
resultsList.place(x=130,y=150, height=80)
data = df["Food_x"].tolist()[:50]
update_results(data)
resultsList.bind("<Double-1>", get_nut_facts)
entryBox.bind("<KeyRelease>", check)

#photo frame
backButton = ck.CTkButton(window, text='Back to Home', height=40, width=160, font=font, text_color="black", fg_color=button_blue, command=lambda: backToHome())
main = tk.Label(nutritionFrame)
camButton = ck.CTkButton(window, text="Take a Pic!", height=80, width=320, font=(font, 30), text_color="black", fg_color=button_blue,  command=lambda: placePhotoFrame())
camButton.place(x=80,y=600)

#after frame
foodLabel = ck.CTkLabel(nutritionFrame, text='Your Food:', height=40, width=160, font=(font, 25, "bold"), text_color="black", corner_radius=8,)
beforeLabel = ck.CTkLabel(nutritionFrame, text='Before:', height=40, width=160, font=(font, 20, "bold"), text_color="black", corner_radius=8,)
afterLabel = ck.CTkLabel(nutritionFrame, text='After:', height=40, width=160, font=(font, 20, "bold"), text_color="black", corner_radius=8,)

afterCalsLabel = ck.CTkLabel(nutritionFrame, text='Calories: '+str(cals), height=40, width=160, font=font, text_color="black", corner_radius=8,)
afterCarbsLabel = ck.CTkLabel(nutritionFrame, text='Carbohydrates: '+str(carbs)+" g", height=30, width=160, font=font, text_color="black", corner_radius=8,)
afterProteinLabel = ck.CTkLabel(nutritionFrame, text='Protein: '+str(protein)+" g", height=30, width=160, font=font, text_color="black", corner_radius=8,)
afterUnsatsLabel = ck.CTkLabel(nutritionFrame, text='Unsaturated Fat: '+str(unsatFats)+" g", height=30, width=160, font=font, text_color="black", corner_radius=8,)
afterSatsLabel = ck.CTkLabel(nutritionFrame, text='Saturated Fat: '+str(satFats)+" g", height=30, width=160, font=font, text_color="black", corner_radius=8,)
afterPolyLabel = ck.CTkLabel(nutritionFrame, text='Polyunsaturated Fat: '+str(polyFats)+" g", height=30, width=160, font=font, text_color="black", corner_radius=8,)
afterSatietyLabel = ck.CTkLabel(nutritionFrame, text='Satiety Index Score: '+str(satiety), height=30, width=160, font=(font,  17, "bold"), text_color="black", corner_radius=8,)

afterlabels = []
afterlabels.append(afterCalsLabel)
afterlabels.append(afterCarbsLabel)
afterlabels.append(afterProteinLabel)
afterlabels.append(afterSatsLabel)
afterlabels.append(afterUnsatsLabel)
afterlabels.append(afterPolyLabel)
afterlabels.append(afterSatietyLabel)

eatBackButton = ck.CTkButton(window, text='Back', height=40, width=160, font=font, text_color="black", fg_color="#BCF4F5", command=lambda: eatBack())
eatButton = ck.CTkButton(window, text="Eat!", height=40, width=160, font=font, text_color="black", fg_color="#BCF4F5", command=lambda: eat())

config_labels_bg(labels, current)
config_satiety_bg(satietyLabel, satiety)

#error msg
errorLabel = ck.CTkLabel(nutritionFrame, text='No nutrition label was identified, please try again.', height=40, width=160, font=font, text_color="red")

window.mainloop()