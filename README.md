# Human Body Level Classification

## Quick Statistics

-   1477 data samples.
-   16 attributes (+1 for classes).
-   4 output classes.

## Attributes Description

-   Gender: Male or female.
-   Age: Numeric value.
-   Height: Numeric value (in meters).
-   Weight: Numeric value (in kilograms).
-   Fam_Hist: Does the family have a history with obesity?
-   H_Cal_Consump: High caloric food consumption.
-   Veg_Consump: Frequency of vegetables consumption.
-   Meal_Count: Average number of meals per day.
-   Food_Between_Meals: Frequency of eating between meals.
-   Smoking: Is the person smoking?
-   Water_Consump: Frequency of water consumption.
-   H_Cal_Burn: Does the body have high calories burn rate?
-   Phys_Act: How often does the person do physical activities?
-   Time_E_Dev: How much time does person spend on electronic devices.
-   Alcohol_Consump: Frequency of alcohols consumption.
-   Transport: Which transports does the person usually use?
-   Body_Level: Class of human body level.

## File Structure
- *Dataset* folder contains the dataset in csv format.
- *Model_No_CV* folder contains the model without cross validation.
  - `model_selection.ipynb` contains the code for model selection.
- *Model_With_CV* folder contains the model with cross validation.
  - `model_selection.ipynb` contains the code for model selection.
- *Preprocess* folder contains the preprocessing code.
  - `preprocess.py` contains the code for preprocessing, used in Cross Validation.
  - `preprocess2.py` contains the code for preprocessing, used in without Cross Validation.