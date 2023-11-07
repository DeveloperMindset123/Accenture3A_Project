# Accenture3A_Project: Detecting The Next Data Breach
This is the shared repository for the Bridge to Studio AI program where we are working on a project from Accenture.

## Currently Working On (October 2023)

Georgina : Data munging and complete data visualizations

Progress: 

(Nov 6) I fixed the Records column (where a non-int value is found, the entry's industry's mean value is used instead. Where there are no available entries with the same industry, the average from the enture dataset is used.) The Years column is now fixed by dividing the records across the range of years, where the breach occurred across multiple years. There are some data visualization graphs now.

Next: I think instead of using the average from the entire dataset, I can use the average from the data breaches in that year alone. I also think there are some duplicate Organizations, which we may want to combine.

(Oct 16) I modified the dataframe for visualization by dealing with ill-formed year values (where the breach occurred across multiple years), but I've realized that the Challenge Advisor's proposed solution may exaggerate data if I copy the records lost. (eg 9 mil records lost in 2019-2020 should not become 9mil records lost in both years). Instead, my next step would be to fix the Records data first (to make them well-formed as integer values), and decide how to split the data across the years that the data breach lasted – most likely dividing across the years, unless otherwise stated in the Records entry.
Question for CA(s), how should I deal with these cases? There're 3 instances of this in the current dataset so its not huge, but should I just divide across the range of years?

Next, I'll do something similar with the ill formed Records. 

Ayan: Merge new datasets

Progress: Added new datasets from Kaggle

Maria: Look into how to implement additional data visualization with the merged dataset.

Progress: 

Urvi: Look into how to implement linear regression models that will be best suitable given our data.

Progress: 

Michelle: Look into some relevant NLP models we can use for this kind of data.

Progress: 
