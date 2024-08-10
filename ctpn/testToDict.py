# import re
#
# addCals, addCarbs, addProtein, addSatFats, addUnsatFats, addPolyFats = 0, 0, 0, 0, 0, 0
# keys = ['cal', 'carb', 'prot', 'sat', 'unsat', 'poly']
# number_pattern = r'[-+]?\d*\.\d+|\d+'
#
# with open('data/resulttext/label.txt', 'r') as file:
#     for line in file:
#         for string in keys:
#             if string in line.lower():
#                 match = re.search(number_pattern, line)
#                 if match:
#                     number = float(match.group())
#                     if string == 'cal':
#                         addCals = int(number)
#                     elif string == 'carb':
#                         addCarbs = number
#                     elif string == 'prot':
#                         addProtein = number
#                     elif string == 'sat' and not line.lower().startswith('un'):
#                         addSatFats = number
#                     elif string == 'unsat':
#                         addUnsatFats = number
#                     elif string == 'poly':
#                         addPolyFats = number
#
# print(addCals, addCarbs, addProtein, addSatFats, addUnsatFats, addPolyFats)

import pandas as pd
from fdc_satnut_pred import predict_satiety
import re
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
    return(predict_satiety(label_dict))