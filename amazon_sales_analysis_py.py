#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# # Importing Amazon Sales Data 

# In[2]:


df = pd.read_csv('Amazon Sales data.csv')
df


# # Dataset Shape

# In[3]:


df.shape


# # Getting info about  non-null and Dtype

# In[4]:


df.info()


# # Finding Null Values

# In[5]:


df.isnull().sum()


# # Classification of Data Types

# In[6]:


df.dtypes == 'object'


# In[7]:


df.dtypes == 'int64'


# In[8]:


df.dtypes == 'float64'


# # Columns of Dataset

# In[9]:


df.columns


# # Converting Date Object to datetime

# In[10]:


df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df.info()


# In[11]:


df.columns


# # Number of Days for Delivery of orders

# In[12]:


df['Delivery days'] = (df['Ship Date']-df['Order Date'])
df


# # Extracting Year,Quarter,Month,Day from Amazon Sales Data

# In[13]:


df['Order Year']=df['Order Date'].dt.year
df['Order Month']=df['Order Date'].dt.month
df['Order Quarter']=df['Order Date'].dt.quarter
df['Order Day']=df['Order Date'].dt.day
df


# # Region Wise Contribution

# In[14]:


plt.figure(figsize = (8,8))
plt.title('Region')
plt.pie(df['Region'].value_counts(), labels=df['Region'].value_counts().index,autopct='%1.1f%%')
plt.show()



# Sub-Saharan Contribution is More than the other region

# # Profit by Region

# In[103]:


region_profit = df.groupby('Region')['Total Profit'].sum()

# Plotting the total profit by region as a pie chart
plt.figure(figsize=(10, 6))
plt.pie(region_profit, labels=region_profit.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
plt.title('Total Profit Distribution by Region')

# Displaying profit values as text in the center of each pie slice
total_profit = region_profit.sum()
plt.text(0, 0, f'Total Profit: ${total_profit}', fontsize=12, color='black', ha='center')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.show()


# # Year Wise Sales

# In[15]:


Year_Sales=df.groupby('Order Year')['Units Sold'].sum()
print(Year_Sales)


# # Top 10 Highest Sales Country

# In[16]:


sales_country = df.groupby(['Country'] , as_index = False)['Units Sold'].sum().sort_values(by='Units Sold',ascending = False)
Top10_sales_country = sales_country.head(10)
print(Top10_sales_country )


# # Year-Month Wise Sales

# In[17]:


monthly_sales = df.groupby(['Order Year', 'Order Month'])['Units Sold'].sum().reset_index()
monthly_sales_sorted = monthly_sales.sort_values(by=['Order Year','Order Month'])
print(monthly_sales_sorted)


# # Year Wise Units Sold in Top Order

# In[18]:


Highest_Sale=df.groupby(['Order Year'],as_index=False)['Units Sold'].sum().sort_values(by='Units Sold',ascending=False)
print(Highest_Sale)


# In[19]:


ax = sns.barplot(x='Order Year',y='Units Sold',data=Highest_Sale)
for bars in ax.containers:
    ax.bar_label(bars)


# From the above graph analyze that in 2012 the Sales is most

# In[20]:


plt.figure(figsize=(8,10))
x=sns.barplot(data=df,x='Sales Channel', y='Units Sold',hue = 'Order Priority')
for bar in x.containers:
    x.bar_label(bar)


# #We can observe that in Offline sales channel the most sale order priority is 'M'
# 
# 
# 
# #we can observe that in Online sales channel the most sale order priority is 'L'

# In[21]:


sales_channel_units_sold = df.groupby('Sales Channel')['Units Sold'].sum()

plt.figure(figsize=(10, 6))
ax = sales_channel_units_sold.plot(kind='bar', color='skyblue')
plt.title('Sales Channel-wise Units Sold')
plt.xlabel('Sales Channel')
plt.ylabel('Units Sold')
plt.xticks(rotation=0, ha='right')  

# Adding text labels on top of each bar
for i in ax.patches:
    ax.text(i.get_x() + i.get_width()/2, i.get_height(), str(int(i.get_height())), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# From the above visualization we can analyze that our sales is more in offline mode as compared to online mode

# # Top10_Sales_Country 

# In[22]:


sales_country = df.groupby(['Country'] , as_index = False)['Units Sold'].sum().sort_values(by='Units Sold',ascending = False)
Top10_sales_country = sales_country.head(10)
print(Top10_sales_country )


# # Day wise Sales 

# In[23]:


Highest_day_sales=df.groupby(['Order Day'],as_index=False)['Units Sold'].sum().sort_values(by='Units Sold',ascending=False)
Top_days_sales = Highest_day_sales
print(Top_days_sales)


# In[24]:


# top_10_States 
top_10_state = df['Country'].value_counts().head(10)
# Plot count of cities by state
plt.figure(figsize=(16, 6))
sns.countplot(data=df[df['Country'].isin(top_10_state.index)], x='Country')
plt.xlabel('country')
plt.ylabel('count')
plt.title('Distribution of  State')
plt.xticks(rotation=45)
plt.show()


# In[25]:


# Prepare data for scatter plot
x_data = df['Order Year']  
y_data = df['Total Profit'] 

# Plot the scatter plot
plt.scatter(x_data, y_data)
plt.xlabel('item type ')  
plt.ylabel('Profit')  
plt.title('Scatter Plot') 
plt.show()


# In[26]:


c=pd.DataFrame(df.groupby('Item Type').sum()['Units Sold'])
print(c)


# In[27]:


v=c.sort_values('Units Sold',ascending=False)
print(v)


# In[28]:


df.groupby('Order Year').sum()


# # Month_Year wise Sales Trends

# In[29]:


df['Order Date'].min()


# In[30]:


df['Order Date'].max()


# In[31]:


df['month_year']=df['Order Date'].apply(lambda x: x.strftime('%Y-%m'))


# In[32]:


sales_trend=df.groupby('month_year').sum()['Units Sold'].reset_index()


# In[33]:


plt.figure(figsize=(16,5))
plt.plot(sales_trend['month_year'],sales_trend['Units Sold'])
plt.xticks(rotation='vertical',size=10)
print(' '*50+'Month_Year Wise Trend')
plt.show()


# # Top10 Item Sales

# In[34]:


Item_sales=pd.DataFrame(df.groupby('Item Type').sum()['Units Sold'])


# In[35]:


Top_sales=Item_sales.sort_values('Units Sold',ascending=False)


# In[36]:


Top_sales[0:10]


# In[37]:


pd.DataFrame(df.groupby('Sales Channel').sum()['Units Sold'])


# In[38]:


df.columns


# In[39]:


Region_sales=pd.DataFrame(df.groupby('Item Type').sum()['Units Sold'])
Top_Region=Region_sales.sort_values('Units Sold',ascending=False)
print(Top_Region)


# In[40]:


Region_sales=pd.DataFrame(df.groupby('Region').sum()['Units Sold'])
print(Region_sales)


# In[41]:


Top_Region=Region_sales.sort_values('Units Sold',ascending=False)
print(Top_Region)


# In[42]:


country_region_profit=pd.DataFrame(df.groupby(['Region','Country']).sum()['Total Profit'])


# In[43]:


country_region_profit.sort_values(['Region','Country'],ascending=False)


# In[44]:


df.groupby('Region').sum()['Units Sold']


# In[45]:


Order_Date = pd.to_datetime(df['Order Date']).dt.date
print(Order_Date)


# In[46]:


Order_Day= pd.to_datetime(df['Order Date']).dt.date
top10sales = df.groupby('Order Date').sum().sort_values('Units Sold', ascending = False)
top10sales = top10sales.reset_index().head(10)
print(top10sales)


# In[47]:


y = top10sales.groupby(['Order Date'])['Units Sold'].sum().nlargest(10)
y.plot.barh()



# From the above graph analyze that the most sales sold on 2013-07-05

# In[48]:


plt.figure(figsize=(18,7))
plt.pie('Units Sold',labels='Order Date',data = top10sales,
        autopct='%1.2f%%',shadow=True,startangle=90)
plt.axis('equal')
plt.title('Contribution Of Sales Amount Among 10 Days')
plt.legend(round(top10sales['Units Sold'],2), loc=7, fontsize = 'x-large')
plt.show()


# In[49]:


Order_Day= pd.to_datetime(df['Order Date']).dt.date
top10sales = df.groupby('Order Date').sum().sort_values('Total Profit', ascending = False)
top10sales = top10sales.reset_index().head(10)
print(top10sales)


# In[50]:


plt.figure(figsize=(18,7))
plt.pie('Total Profit',labels='Order Date',data = top10sales,
        autopct='%1.2f%%',shadow=True,startangle=90)
plt.axis('equal')
plt.title('Contribution Of Sales Amount Among 10 Days')
plt.legend(round(top10sales['Total Profit'],2), loc=7, fontsize = 'x-large')
plt.show()


# In[51]:


Order_Day= pd.to_datetime(df['Order Date']).dt.date
top10sales = df.groupby('Order Date').sum().sort_values('Total Revenue', ascending = False)
top10sales = top10sales.reset_index().head(10)
print(top10sales)


# In[52]:


plt.figure(figsize=(18,7))
plt.pie('Total Revenue',labels='Order Date',data = top10sales,
        autopct='%1.2f%%',shadow=True,startangle=90)
plt.axis('equal')
plt.title('Contribution Of Sales Amount Among 10 Days')
plt.legend(round(top10sales['Total Revenue'],2), loc=7, fontsize = 'x-large')
plt.show()


# In[53]:


df.groupby('Country').sum().sort_values('Total Profit',ascending=False).head(10)



# # Region Wise Total Profit

# In[54]:


region_wise = df.groupby(['Region'])['Total Profit'].sum()
region_wise.plot.barh()



# From the above visulization we can analyze that the Sub-Saharan Africa is the Most Profitable region

# In[55]:


top_country_profit = df.groupby("Country")['Total Profit'].sum().head(10)
print(top_country_profit)


# In[56]:


Region_sales=pd.DataFrame(df.groupby('Item Type').sum()['Units Sold'])
Top_Region=Region_sales.sort_values('Units Sold',ascending=False)
print(Top_Region)


# In[96]:


plt.figure(figsize=(14,10)) 
statewise = df.groupby(['Item Type'])['Total Profit'].sum().nlargest(100) 
statewise.plot.barh() # h for horizontal
for index, value in enumerate(statewise):
    plt.text(value, index, str(int(value)), ha='left', va='center')

plt.title('Top 100 Item Types by Total Profit')
plt.xlabel('Total Profit')
plt.ylabel('Item Type')
plt.tight_layout()
plt.show()


# hence From the above visulization cosmetic product is the most Profitable product from all the item

# In[58]:


Year_wise_Sales=df.groupby('Order Year').sum().reset_index()
sns.catplot(y = 'Units Sold', x = 'Order Year', data = df, palette='Reds',kind="bar")
plt.xlabel('Year')
plt.ylabel('Units Sold')
plt.title('Yearly Sales')
Year_wise_Sales[['Order Year', 'Units Sold']]


# In[59]:


year_monthwise_sales = df.groupby(['Order Year','Order Month']).sum().reset_index()
print(year_monthwise_sales)


# In[60]:


sns.relplot(x ='Order Month',y = 'Units Sold', data=year_monthwise_sales,height=5,
            kind = 'line', aspect = 0.4, col = 'Order Year')
plt.xlabel('Order Month')
plt.ylabel('Units Sold')
print( ' '*40  +'Year Wise Month Trends')


# In[61]:


sns.catplot(y = 'Units Sold', x = 'Order Month', data = year_monthwise_sales, aspect=2.0,palette='turbo',kind="bar",
            col='Order Year', col_wrap=3)


# In[62]:


sns.histplot(year_monthwise_sales['Units Sold'], kde = True)


# # Monthly-Wise Record

# In[63]:


Monthwise_sales=df.groupby(['Order Year','Order Month','Order Day']).sum().reset_index()


# In[64]:


Monthwise_sales.describe()


# In[65]:


monthwise_sales=df.groupby(['Order Year', 'Order Month'])['Units Sold'].sum().reset_index()


# In[66]:


plt.figure(figsize=(15, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == year]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data, label=str(year))

    # Annotate each point with units sold value
    for index, row in year_data.iterrows():
        plt.annotate(str(row['Units Sold']), (row['Order Month'], row['Units Sold']), textcoords="offset points", xytext=(0,5), ha='center')

plt.title('Monthly Sales on Amazon Year-wise')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Year')
plt.tight_layout()
plt.show()


# # Analyzing Month Trend Sales with each Year Separately

# In[67]:


plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == 2010]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data)

plt.title('Monthly Sales on Amazon 2010')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Year')
plt.show()


# In[68]:


plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == 2011]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data)

plt.title('Monthly Sales on Amazon 2011')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[69]:


plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == 2012]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data)

plt.title('Monthly Sales on Amazon 2012')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.show()


# In[70]:


plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == 2013]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data)

plt.title('Monthly Sales on Amazon 2013')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[71]:


plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == 2014]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data)

plt.title('Monthly Sales on Amazon 2014')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[72]:


plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == 2015]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data)

plt.title('Monthly Sales on Amazon 2015')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.show()


# In[73]:


plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == 2016]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data)

plt.title('Monthly Sales on Amazon 2016')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[74]:


plt.figure(figsize=(6, 4))
sns.set_style("whitegrid")

# Plot each year separately
for year in monthwise_sales['Order Year'].unique():
    year_data = monthwise_sales[monthly_sales['Order Year'] == 2017]
    sns.lineplot(x='Order Month', y='Units Sold', data=year_data)

plt.title('Monthly Sales on Amazon 2017')
plt.xlabel('Month')
plt.ylabel('Units Sold')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# # Year-Wise Sales Trends

# In[75]:


yearly_sales = df.groupby('Order Year')['Units Sold'].sum().reset_index()
print(yearly_sales)


# In[76]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Order Year', y='Units Sold', data=yearly_sales, palette='viridis')
plt.title('Yearly Total Sales on Amazon')
plt.xlabel('Year')
plt.ylabel('Total Sales Amount ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[77]:


plt.figure(figsize=(10, 6))
sns.lineplot(x='Order Year', y='Units Sold', data=yearly_sales, marker='o', color='skyblue')
plt.title('Yearly Total Sales on Amazon')
plt.xlabel('Year')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()







# In[78]:


plt.figure(figsize=(8, 4))
plt.pie(yearly_sales['Units Sold'], labels=yearly_sales['Order Year'], autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Yearly Total Sales Distribution on Amazon')
plt.tight_layout()
plt.show()


# In[79]:


yearly_sales = df.groupby('Order Year')['Total Profit'].sum().reset_index()
print(yearly_sales)


# In[80]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Order Year', y='Total Profit', data=yearly_sales, palette='viridis')
plt.title('Yearly Total Sales on Amazon')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[81]:


plt.figure(figsize=(8, 4))
plt.pie(yearly_sales['Total Profit'], labels=yearly_sales['Order Year'], autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Yearly Total Profit Distribution on Amazon')
plt.tight_layout()
plt.show()


# In[82]:


df.columns


# In[83]:


yearly_sales = df.groupby('Order Year')['Total Profit'].sum().reset_index()
print(yearly_sales)


# In[84]:


plt.figure(figsize=(14, 10)) 
statewise = yearly_sales.groupby(['Order Year'])['Total Profit'].sum()
statewise.plot.barh() 

for index, value in enumerate(statewise):
    plt.text(value, index, str(round(value, 2)))

plt.xlabel('Total Profit')
plt.ylabel('Order Year')
plt.title('Yearly Total Profit')
plt.show()



# IN 2012 We got The Most Profit It was the Most Profitable Year

# In[85]:


yearly_item_sales = df.groupby(['Order Year', 'Item Type']).sum()['Units Sold'].reset_index()
print(yearly_item_sales)


# In[86]:


plt.figure(figsize=(10, 6))
sns.barplot(data=yearly_item_sales, x='Order Year', y='Units Sold', hue='Item Type')
plt.title('Most Sold Product Each Year')
plt.xlabel('Year')
plt.ylabel('Units Sold')
plt.legend(title='Product')
plt.show()


# From the above visualization we analyze the most sale product year wise as given below
# 
# 2010 = 'Clothes'
# 2011 = 'Beverages'
# 2012 = 'Personal Care'
# 2013 = 'Cosmetics'
# 2014 = 'Beverages'
# 2015 = 'Clothes'
# 2016 = 'Cosmetics'
# 2017 = 'Personal Care'

# In[87]:


yearly_sales = df.groupby(['Order Year', 'Item Type'])['Units Sold'].sum().reset_index()

# Find the most sold item for each year
most_sold_per_year = yearly_sales.loc[yearly_sales.groupby('Order Year')['Units Sold'].idxmax()]

# Print the results
print(most_sold_per_year)


# In[88]:


df.columns


# # Forecasting Sales And Profit

# In[89]:


from prophet import Prophet


# In[90]:


df['date'] =pd.to_datetime(df['Order Date'])


# In[91]:


sales_data = df.rename(columns={'Order Date': 'ds', 'Units Sold': 'y'})


# In[92]:


model = Prophet()


# In[93]:


model.fit(sales_data)


# In[94]:


profit_data = df.rename(columns={'Order Date': 'ds', 'Total Profit': 'y'})
model = Prophet()
model.fit(profit_data)
future = model.make_future_dataframe(periods=365*4)
forecast = model.predict(future)
fig = model.plot(forecast)


# In[95]:


profit_data = df.rename(columns={'Order Date': 'ds', 'Units Sold': 'y'})
model = Prophet()
model.fit(profit_data)
future = model.make_future_dataframe(periods=365*4)
forecast = model.predict(future)
fig = model.plot(forecast)


# In[100]:


df.columns


# In[102]:


region_profit = df.groupby('Region')['Total Profit'].sum()

# Plotting the total profit by region as a pie chart
plt.figure(figsize=(10, 6))
plt.pie(region_profit, labels=region_profit.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
plt.title('Total Profit Distribution by Region')

# Displaying profit values as text in the center of each pie slice
total_profit = region_profit.sum()
plt.text(0, 0, f'Total Profit: ${total_profit}', fontsize=12, color='black', ha='center')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.show()


# In[ ]:




