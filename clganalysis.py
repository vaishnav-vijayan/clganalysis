#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt


# In[19]:


data = pd.read_csv('Engineering Student Count Details (College wise - Branch wise).csv')

data['Zone'] = data['Zone'].fillna(method='ffill')

data['Group'] = data['Group'].fillna(method='ffill')

data = data[(data['Category Name'] != 'Architecture') & (data['Category Name'] != 'Affiliated Autonomous')]


#copy data to xdata
xdata = data.copy()
#delete branch from xdata
del xdata['branch']

#merge rows with same college code
xdata = xdata.groupby(['College Code','College Name','Zone','Group','Category Name','University','District']).sum().reset_index()
#delete rows with less than 20 total students or 1st year students
condition1 = xdata['Total No of students'] <= 20
condition2 = xdata['1st year'] <= 20
xdata = xdata[condition1 | condition2]


#delete rows from data where college code is in xdata
data = data[~data['College Code'].isin(xdata['College Code'])]


# In[20]:


one_a_data = data.copy()
del one_a_data['branch']
one_a_data = one_a_data.groupby(['College Code','College Name','Zone','Group','Category Name','University','District']).agg({'1st year':'sum', 'Total No of students': 'sum'}).reset_index()
one_a_data = one_a_data.groupby('Zone').agg({'College Code': 'count',  'Total No of students': 'sum', '1st year':'sum'}).reset_index()
one_a_data = one_a_data.rename(columns={'College Code': 'Number of Colleges'})
print("\033[1mNumber of colleges in zone wise with Total No of students and total 1st year students\033[0m")
one_a_data


# In[21]:


sns.set(style = 'whitegrid')
fig, ax = plt.subplots(figsize = (10,6))

sns.barplot(x='Zone', y='Number of Colleges', data = one_a_data, color = 'red', alpha = 0.7, label = 'Total Colleges')
sns.barplot(x='Zone', y='1st year', data = one_a_data, color = 'green', alpha = 0.7, label = 'Sum of 1st Year Students')

ax.set_xlabel('Zone', fontsize = 14)
ax.set_ylabel('Count', fontsize = 14)
ax.set_title('Number of Colleges and number of first year students by zone', fontsize = 16)
plt.xticks(rotation = 90)
plt.legend(loc = 'upper right')

for i, bar in enumerate(ax.containers):
    ax.bar_label(bar, label_type = 'edge', labels = one_a_data.iloc[:, i+1], fontsize = 11)
plt.show()


# In[22]:


data.loc[data['branch'].isnull(), 'branch'] = 'Others'
data.loc[:, 'Disciplines'] = 'OTHER'


data.loc[data['branch'].str.contains('computer', case = False), 'Disciplines'] = 'CS'
data.loc[data['branch'].str.contains('Artificial Intelligence', case = False), 'Disciplines'] = 'CS'

data.loc[data['branch'].str.contains('information', case = False), 'Disciplines'] = 'IT'

data.loc[data['branch'].str.contains('mech', case = False), 'Disciplines'] = 'MECH'
data.loc[data['branch'].str.contains('auto', case = False), 'Disciplines'] = 'MECH'

data.loc[data['branch'].str.contains('communication', case = False), 'Disciplines'] = 'ECE'
data.loc[data['branch'].str.contains('electronics', case = False), 'Disciplines'] = 'ECE'

data.loc[data['branch'].str.contains('electrical', case = False), 'Disciplines'] = 'EEE'

data.loc[data['branch'].str.contains('civil', case = False), 'Disciplines'] = 'CIVIL'

data.to_csv('my_dataframe.csv', index=False)



# In[23]:


one_b_data = data.groupby(['Zone', 'Disciplines']).size().reset_index(name='count')


# In[24]:


pivoted = one_b_data.pivot(index='Zone', columns='Disciplines', values='count').fillna(0)

ax = pivoted.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_xlabel('Zone')
ax.set_ylabel('Count')
ax.set_title('Number of Disciplines by Zone', fontsize = 16)


for i, patch in enumerate(ax.patches):
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_y() + patch.get_height() / 2
    count = int(patch.get_height())
    ax.text(x, y, count, ha='center', va='center')

plt.show()


# In[25]:


one_c_data = data.groupby('Zone').agg({'College Name': 'nunique', 'Total No of students': 'sum'})

# Compute the maximum and minimum number of colleges and students
max_colleges = one_c_data['College Name'].idxmax()
max_students = one_c_data['Total No of students'].idxmax()
min_colleges = one_c_data['College Name'].idxmin()
min_students = one_c_data['Total No of students'].idxmin()

# Compute the average number of colleges across zones
avg_colleges = one_c_data['College Name'].mean()

# Print the results
print("\033[1mMaximum and minimun stats\033[0m")
print(f"Zone with maximum number of colleges: {max_colleges}")
print(f"Zone with maximum number of students: {max_students}")
print(f"Zone with minimum number of colleges: {min_colleges}")
print(f"Zone with minimum number of students: {min_students}")
print(f"Average number of colleges across zones: {avg_colleges}")


# In[26]:


one_d_data = one_c_data['Total No of students'] / one_c_data['College Name']
print("\033[1mStudent density by zone\n\033[0m")
print(one_d_data)

# Display student density as a bar chart
ax = one_d_data.plot(kind='bar', figsize=(10,6))
plt.xlabel('Zone')
plt.ylabel('Student Density')
plt.title('Student Density by Zone')

for i, bar in enumerate(ax.containers):
    ax.bar_label(bar, label_type='edge', labels=[f"{val:.2f}" for val in one_d_data], fontsize=11)

plt.show()


# In[27]:


zones = data['Zone'].unique()

print("\033[1mHistogram for number of students in each zone(bin size = 20)\033[0m")
def plot_graph(zone):
    zone_data = data[data['Zone'] == zone]
    plt.figure(figsize=(12, 6))
    plt.hist([zone_data['Total No of students'], zone_data['1st year']], bins=20, alpha=0.5, label=['Total Students', '1st Year'])
    plt.title(f'Histogram for Zone {zone}')
    plt.xlabel('Number of Students')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

zone_dropdown1 = widgets.Dropdown(options=zones, description='Select Zone:')
output1 = widgets.Output()

def on_change1(change):
    if change['type'] == 'change' and change['name'] == 'value':
        with output1:
            output1.clear_output()
            plot_graph(change['new'])

zone_dropdown1.observe(on_change1)

display(zone_dropdown1)
display(output1)


# In[28]:


two_b_data = data.groupby(['College Name', 'Zone', 'Disciplines']).agg({'Total No of students':'sum'}).reset_index()


# In[29]:


# pivot the data to create a stacked bar chart
pivot_data = two_b_data.pivot(index='College Name', columns='Disciplines', values='Total No of students')

print("\033[1mNumber of students by department in each college\033[0m")
# define a function to update the chart based on the selected zone
def update_chart(zone_name):
    zone_data = two_b_data.loc[two_b_data['Zone'] == zone_name]
    pivot_data = zone_data.pivot(index='College Name', columns='Disciplines', values='Total No of students')
    pivot_data.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Total Number of Students by Department and College for Zone ' + zone_name)
    plt.xlabel('College Name')
    plt.ylabel('Number of Students')
    plt.show()



zone_dropdown2 = widgets.Dropdown(options=zones, description='Select Zone:')
output2 = widgets.Output()

def on_change22(change):
    if change['type'] == 'change' and change['name'] == 'value':
        with output2:
            output2.clear_output()
            update_chart(change['new'])

zone_dropdown2.observe(on_change22)

display(zone_dropdown2)
display(output2)


# 

# In[30]:


two_c_data = data.groupby(['Zone', 'Disciplines']).agg({'Total No of students':'sum', '1st year':'sum'}).reset_index()


# In[31]:


print("\033[1mTotal number of students and first year students in a discipline by zone\033[0m")
# Define a function to update the histogram based on the selected zone
def update_histogram(zone):
    # Filter the data by the selected zone
    zone_data = two_c_data[two_c_data['Zone'] == zone]
    
    # Group the data by discipline and sum the total number of students
    discipline_data = zone_data.groupby('Disciplines')['Total No of students'].sum()
    
    # Create a bar plot with the number of students on the y-axis and the discipline on the x-axis
    plt.bar(discipline_data.index, discipline_data.values)
    
    # Add labels to the plot
    plt.xlabel('Discipline')
    plt.ylabel('Number of Students')
    plt.title(f'Number of Students in Each Discipline for Zone {zone}')
    
    # Show the plot
    plt.show()
    
def update_histogram1(zone):
    # Filter the data by the selected zone
    zone_data = two_c_data[two_c_data['Zone'] == zone]
    
    # Group the data by discipline and sum the total number of students
    discipline_data = zone_data.groupby('Disciplines')['1st year'].sum()
    
    # Create a bar plot with the number of students on the y-axis and the discipline on the x-axis
    plt.bar(discipline_data.index, discipline_data.values)
    
    # Add labels to the plot
    plt.xlabel('Discipline')
    plt.ylabel('Number of Students')
    plt.title(f'Number of 1st year Students in Each Discipline for Zone {zone}')
    
    # Show the plot
    plt.show()

zone_dropdown3 = widgets.Dropdown(options=zones, description='Select Zone:')
output3 = widgets.Output()

def on_change3(change):
    if change['type'] == 'change' and change['name'] == 'value':
        with output3:
            output3.clear_output()
            update_histogram(change['new'])
            update_histogram1(change['new'])

zone_dropdown3.observe(on_change3)

display(zone_dropdown3)
display(output3)


# In[32]:


print("\033[1mNumber of students and first year students in a discipline by College wise\033[0m")

three_data = data.groupby(['College Name', 'Zone', 'Disciplines']).agg({'Total No of students':'sum', '1st year':'sum'}).reset_index()
# create dropdown widgets for selecting zone and college
zone_dropdown = widgets.Dropdown(options=three_data['Zone'].unique(), description='Zone')
college_dropdown = widgets.Dropdown(description='College')

# define a function to update the college dropdown options based on the selected zone
def update_college_options(*args):
    selected_zone = zone_dropdown.value
    college_dropdown.options = three_data[three_data['Zone'] == selected_zone]['College Name'].unique()

# call the update_college_options function when the zone dropdown value changes
zone_dropdown.observe(update_college_options, 'value')

# define a function to update the histogram based on the selected college
def update_histogram10(*args):
    selected_college = college_dropdown.value
    college_data = three_data[three_data['College Name'] == selected_college]
    discipline_data = college_data.groupby('Disciplines').agg({'Total No of students':'sum', '1st year':'sum'})
    
    # clear the previous plot
    output10.clear_output(wait=True)
    
    # create a new plot
    with output10:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        
        # plot the total number of students
        total_bars = ax[0].bar(discipline_data.index, discipline_data['Total No of students'])
        ax[0].set_xlabel('Disciplines')
        ax[0].set_ylabel('Total Number of students')
        ax[0].set_title('Number of students in each discipline at ' + selected_college)
        
        # add text annotations for total number of students
        for i, bar in enumerate(total_bars):
            ax[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), int(bar.get_height()), ha='center', va='bottom')
        
        # plot the number of 1st year students
        firstyear_bars = ax[1].bar(discipline_data.index, discipline_data['1st year'])
        ax[1].set_xlabel('Disciplines')
        ax[1].set_ylabel('Number of 1st year students')
        ax[1].set_title('Number of 1st year students in each discipline at ' + selected_college)
        
        # add text annotations for number of 1st year students
        for i, bar in enumerate(firstyear_bars):
            ax[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), int(bar.get_height()), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# initialize the output widget
output10 = widgets.Output()

# define a function to handle changes to the college dropdown value
def on_change10(change):
    if change.get('type') == 'change' and change.get('name') == 'value':
        update_histogram10(change['new'])

# call the on_change10 function when the college dropdown value changes
college_dropdown.observe(on_change10)

# display the zone and college dropdown widgets
display(zone_dropdown)
display(college_dropdown)

# display the output widget
display(output10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




