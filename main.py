import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df=sns.load_dataset("tips").dropna()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.isnull().sum()

df.head()

"""# 2. **Basic Plotting with Seaborn**

## **Categorical Plots**:

### barplot():
"""

# Barplot arguments:
# x: Categorical variable for grouping
# y: Numerical variable for calculations
# estimator: Function applied to y (e.g., mean)
# ci: Confidence interval calculated via bootstrapping -> generates sampling distribution -> uses 2.5% and 97.5% percentiles for default 95% CI
# hue: Additional categorical variable for further grouping


fig, ax = plt.subplots(1, 3, figsize=(12, 3))

sns.barplot(x="time",y="tip",data=df, ax=ax[0])
ax[0].set_title("mean tips for each time")

sns.barplot(x="time",y="tip",estimator="std",data=df, ax=ax[1])# u can use max or any describtion
ax[1].set_title("std tips for each time")

sns.barplot(x="time",y="tip",hue="sex",ci=False,data=df, ax=ax[2])
ax[2].set_title("mean tips for each time for each sex")


plt.tight_layout()

# Display the plots
plt.show()

"""### countplot():"""

# countplot is only counting the number of each category (its like hist but for categorical features)
fig, ax = plt.subplots(1, 3, figsize=(12, 3))

sns.countplot(x="time",data=df, ax=ax[0])
ax[0].set_title("count of each time")

sns.countplot(y="time",data=df, palette="dark",ax=ax[1])
ax[1].set_title("Hort count plot")

sns.countplot(x="time",data=df,hue="sex", ax=ax[2])
ax[2].set_title("count of each time")

plt.tight_layout()

# Display the plots
plt.show()

"""### boxplot():"""

# boxplot  shows the distribution of the numerical variable grouped by a categorical variable.
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# Boxplot expects a numerical column to plot its distribution.
sns.boxplot(x="tip", data=df, ax=ax[0,0])
ax[0,0].set_title("Distribution of Tips")  # Title for the first plot

# You can add a categorical column to split the data into groups.
sns.boxplot(y="tip", x="sex", data=df, ax=ax[0,1])  # Splitting 'tip' by the 'sex' category.
ax[0,1].set_title("Tips by Gender")  # Title for the second plot

# Boxplot can be further split by adding a 'hue' parameter to compare more groups.
sns.boxplot(y="total_bill", x="sex", hue="time", data=df, ax=ax[1,0])
ax[1,0].set_title("Total Bill by Gender and Time")  # Title for the third plot

fig.delaxes(ax[1,1])
# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plots
plt.show()

"""### violinplot():"""

# violin plot is a mix of boxplot and kde plots

sns.violinplot(x="sex",y="total_bill",palette="dark",hue="time",data=df)
plt.title ("total bills for each gender")
plt.show()

"""##    **Distribution Plots**:

### histplot():
"""

#x="tip": Specifies the column (tip) from the DataFrame df to be plotted on the x-axis.
#bw_method=10: Sets the bandwidth for the kernel, controlling the smoothness of the KDE curve. A higher value results in a smoother curve.
#fill=True: Fills the area under the KDE curve, making the plot visually appealing.
#data=df: Indicates the DataFrame (df) that contains the data.
# kda after summing kernals the valuse is normalized
# This code generates a bivariate KDE plot that shows the joint distribution of the tip and total_bill variables

fig,ax = plt.subplots(1,2,figsize=(8,5))
sns.kdeplot(x="tip",bw_method=10,fill=True,data=df,ax=ax[0])
sns.kdeplot(x="tip",y="total_bill",bw_method=10,fill=True,ax=ax[1],data=df)
plt.show()

"""### Kdeplot():"""

# Create the subplots (3 rows and 3 columns)
fig, ax = plt.subplots(3, 3, figsize=(15, 12))

# First row of plots
sns.histplot(x="tip", data=df, ax=ax[0, 0])
ax[0, 0].set_title("Histogram of Tips")

sns.histplot(y="tip", data=df, ax=ax[0, 1])
ax[0, 1].set_title("Histogram of Tips (Y-axis)")

sns.histplot(x="tip", kde=True, data=df, ax=ax[0, 2])
ax[0, 2].set_title("Histogram with KDE of Tips")

# Second row of plots
sns.histplot(x="tip", stat="density", data=df, ax=ax[1, 0])
ax[1, 0].set_title("Density Plot of Tips")

sns.histplot(x="tip", stat="probability", data=df, ax=ax[1, 1])
ax[1, 1].set_title("Probability Plot of Tips")

sns.histplot(x="tip", y="total_bill", data=df, ax=ax[1, 2])
ax[1, 2].set_title("Joint Plot of Tip vs. Total Bill")

# Third row of plots (one plot and two empty subplots)
sns.histplot(x="total_bill",hue="sex", data=df, ax=ax[2, 0])
ax[2, 0].set_title("Histogram of Total Bill For Different sex")

sns.histplot(x="tip", data=df,color="black", ax=ax[2, 1])
ax[0, 0].set_title("Histogram of Tips using Color")

sns.histplot(x="tip", y="total_bill",hue="sex", palette="bone",data=df, ax=ax[2, 2])
ax[1, 2].set_title("Joint Plot of Tip vs. Total Bill")


# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

"""##   **Relational Plots**:

## scatterplot():
"""

# Scatter plot visualizes the relationship between two numerical features, where each point represents a pair of values (x, y).
# Insights: Scatter plots can reveal correlations, patterns, or clusters in the data, helping to identify relationships or potential outliers.
# Parameters:
# x: Values of the first feature (plotted on the x-axis)
# y: Values of the second feature (plotted on the y-axis)
# s: Size of the scatter plot points
# hue: Optional parameter to color points based on a third categorical feature

fig,ax=plt.subplots(1,2,figsize=(8,5))
sns.scatterplot(x="tip",y="total_bill",hue="time",s=50,ax=ax[0],data=df)
sns.scatterplot(x="tip",y="total_bill",hue="time",s=100,ax=ax[1],data=df)
plt.tight_layout()
plt.show()

"""## lineplot():"""

# Line plot visualizes the trend or relationship between two numerical features over a continuous variable, often time.
# Insights: Line plots can show trends, changes over time, and the consistency or variability of data, making it easier to observe patterns or seasonality.
# Parameters:

# x: Values of the feature to be plotted on the x-axis (typically represents time or another continuous variable)

# y: Values of the feature to be plotted on the y-axis (shows the trend or change over the x variable)

# hue: Optional parameter to differentiate lines by coloring them based on a third categorical feature

# ci: Optional parameter to display a confidence interval around the line, representing the uncertainty or variability of the data.
#     The confidence interval is typically calculated using bootstrapping, which estimates the variability of the data by resampling
#     the data with replacement multiple times and calculating the mean and standard deviation for each resample.

# n_boot: Optional parameter to specify the number of bootstrap resamples used to calculate the confidence interval (default is 1000).
#         Increasing this number can lead to more stable estimates but may also increase computation time.

# estimator: Optional parameter to specify the statistical function used to estimate the trend line (default is the mean).
#            Common choices include functions like np.mean, np.median, or custom functions to calculate the desired statistic.


fig,ax=plt.subplots(1,3,figsize=(12,5))
sns.lineplot(x="size",y="total_bill",ci=None,hue="time",ax=ax[0],data=df)
sns.lineplot(x="size",y="total_bill",hue="time",ax=ax[1],data=df)
sns.lineplot(x="size",y="total_bill",hue="time",estimator="median",ax=ax[2],data=df)

plt.tight_layout()
plt.show()

"""# 3. **Customization and Styling**

##  **Themes and Aesthetics**:

- **`set()`**:
- **`set_style()`**: Explore how to change the look and feel of your plots using Seaborn’s themes.
- **`set_palette()`**: Customize color palettes to match your data visualization needs.
"""

# set()
# Key Options:
#     style: Chooses the general look of the plot (e.g., white, dark, whitegrid, darkgrid, ticks).
#     context: Adjusts the size of plot elements like labels and lines, depending on whether you’re printing or presenting (e.g., paper, notebook, talk, poster).
#     palette: Decides the color scheme used in the plot (e.g., vibrant or muted colors) (e.g., deep, muted, bright, dark, colorblind).



sns.set(style="whitegrid", palette="muted", context="talk")
sns.barplot(x="time", y="total_bill", data=df)
plt.show()

sns.set(style="darkgrid", palette="Blues", context="paper")
sns.barplot(x="time", y="total_bill", data=df)
plt.show()

sns.set(style="ticks", palette="bright", context="notebook")
sns.barplot(x="time", y="total_bill", data=df)
plt.show()

# set_style() is designed for changing the style only  same as set_palette()

sns.set_style("dark")
sns.barplot(x="time", y="total_bill", data=df)
plt.show()

sns.set_palette("dark")
sns.barplot(x="time", y="total_bill", data=df)
plt.show()

"""  ## **Legend Customization**: Adjust legend position, titles, and labels to improve plot readability."""

# legend parameters :
# 1. loc : Specifies the location of the legend in the plot (best,upper right ,upper left ,lower right,lower left,....).
# 2. title : Adds a title to the legend box, which helps explain what the legend is about
# 3. fontsize: Controls the font size of the legend labels. elso title_fontsize
# 4. labels: Manually sets the labels for the legend.
# 5. frameon: Controls whether the legend has a border (frame) around it.

sns.set(style="white",palette="deep",context="notebook")
sns.boxplot(x="sex",y="total_bill",hue="time",data=df)
plt.legend(loc="best",title="Time",fontsize=8,frameon=False)
plt.show()

"""## Adding Vertical And Horizontal Lines"""

# we use axvline for vertical and axhline for horizontal lines
# parameters :
# 1. x,y: The x,y-coordinate where the vertical line should be drawn.
# 2. color: The color of the line (e.g., 'red', 'blue', etc.).
# 3. linestyle: The style of the line (e.g., '--' for dashed, '-' for solid).
# 4. linewidth: The width of the line.
# 5. label : name of the line
sns.histplot(x="total_bill",data=df)
plt.axvline(x=df["total_bill"].mean(),color="red",linestyle="-",label="Mean",linewidth=2)
plt.axvline(x=df["total_bill"].median(),color="blue",linestyle="-",label="Median",linewidth=2)

plt.legend()
plt.show()

"""# 4. **Advanced Plotting Techniques**

##  Facet Grids:
"""

# FacetGrid: Creates a grid layout to visualize multiple plots across different subsets of the data.
# You have three steps to plot a FacetGrid:
# 1. Generate the FacetGrid object.
# 2. Apply the map_dataframe function to plot data on the grid.
# 3. Customize the style for clarity and aesthetics.

# Parameters:
    # For FacetGrid:
        # data: The DataFrame containing your data.
        # col: Variable that defines how the data is split across columns of the grid.
        # row: Variable that defines how the data is split across rows of the grid.
        # hue: Variable used to add color to different subsets of data within each subplot.

    # For map_dataframe:
        # func: The plotting function to apply (e.g., sns.scatterplot).
        # **kwargs: Additional keyword arguments passed to the plotting function (e.g., x, y, size).

    # For style:
        # set_titles: Adds titles to each facet, usually based on the subset of data.
        # set_axis_labels: Sets common x and y axis labels for the entire grid.

plt.figure(figsize=(12,8))
g=sns.FacetGrid(data=df,col="size",row="sex",hue="time")
g.map_dataframe(sns.kdeplot,x="total_bill")
g.set_axis_labels("total bill")
g.set_titles("{col_name} | {row_name}")
g.add_legend()

plt.show()

"""##  Heatmaps:"""

# Generate a heatmap to visualize the relationship between two categorical variables
# Parameters:
# data: The DataFrame or 2D array containing the data to visualize.
# annot: If True, write the data value in each cell.
# fmt: String formatting code to use when adding annotations (e.g., ".2f" for 2 decimal places).
# cmap: Color map to use for the heatmap (e.g., "coolwarm", "viridis", "YlGnBu").
# linewidths: Width of the lines that divide each cell in the heatmap.
# cbar: If True, display the color bar showing the scale of the heatmap.
# vmax: The maximum value for the color scale. Values higher than this will be clipped to the maximum color.
# vmin: The minimum value for the color scale. Values lower than this will be clipped to the minimum color.


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sns.heatmap(data=df.drop(columns=["sex","time","smoker","day"]).corr(), annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar=True, vmax=1, vmin=0, ax=ax1)
ax1.set_title("Heatmap with Color Bar", fontsize=16)
ax1.set_xlabel("X-axis Label", fontsize=14)
ax1.set_ylabel("Y-axis Label", fontsize=14)

sns.heatmap(data=df.drop(columns=["sex","time","smoker","day"]).corr(), annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, cbar=False, vmax=1, vmin=0, ax=ax2)
ax2.set_title("Heatmap without Color Bar", fontsize=16)
ax2.set_xlabel("X-axis Label", fontsize=14)
ax2.set_ylabel("Y-axis Label", fontsize=14)

# Display the plots
plt.tight_layout()
plt.show()

"""##  Pairwise Relationships:

"""

df=sns.load_dataset("titanic").drop(columns=["alone","sibsp","parch","adult_male","survived"]).dropna()
df.head()

df.info()

"""### pairplot()

"""

# pairplot: A PairPlot is a powerful visualization tool that generates a matrix of plots for each pair of variables in a dataset. This allows for a comprehensive examination of the relationships and distributions between multiple variables simultaneously. It's particularly useful in exploratory data analysis (EDA) to identify patterns, correlations, and outliers.

# insights:
# - PairPlots are valuable for identifying trends, correlations, and outliers between variables.
# - They help in understanding the overall distribution of data across variables, making it easier to detect skewness or any unusual distributions.
# - PairPlots can also reveal multi-dimensional relationships that might not be evident when examining variables individually.

# notes:
    # By default, the diagonal plots are histograms (`histplot`), which display the distribution of each individual variable, making it easy to see the spread and central tendency.
    # The off-diagonal plots are scatterplots (`scatterplot`), showing the relationship between each pair of variables, which can highlight correlations or lack thereof.
    # You can change the diagonal plots using `diag_kind` (e.g., `kde` for a smooth curve representing the distribution) and the off-diagonal plots using `kind` (e.g., `reg` for adding a regression line).
    # PairPlot treats boolean features as numerical (0 and 1), so it includes them in the plot matrix, which can be useful for visualizing the impact of categorical variables.
    # If your dataset has many features, the resulting PairPlot can become overwhelming. In such cases, consider selecting a subset of relevant variables to focus the analysis.
    # PairPlot can be slow to generate for large datasets due to the number of plots it creates. Preprocessing your data to reduce its size or using a sample can improve performance.
    #  x_vars= Selects specific variables to display on the x-axis
    #  y_vars= Selects specific variables to display on the y-axis
sns.set_palette("dark")
sns.pairplot(df,hue="sex")
plt.show()

sns.set_palette("dark")
sns.pairplot(df,hue="sex",diag_kind="kde",kind="reg")
plt.show()

"""### jointplot()"""

# jointplot: A JointPlot is a powerful visualization tool that focuses on the relationship between two variables.

# It combines a relational plot (joint_kws) with univariate plots (like histograms or KDEs) (marginal_kws)

# paramaters:
    #  x=  Selects the variable to display on the x-axis
    #  y=  Selects the variable to display on the y-axis
    #  kind="scatter": Specifies the type of plot for the central area, default is scatterplot
    #  hue="species": Colors the data points based on the specified categorical variable
    #  marginal_kws=dict(bins=15, fill=True): Customizes the marginal plots
    #  joint_kws=dict(alpha=0.7): Customizes the joint plot, here setting the transparency of the points

sns.jointplot(x="age",y="fare",hue="sex",data=df)

"""# 5. **Saving and Exporting Plots**

## Save Plots: save your Seaborn plots in different file formats (e.g., PNG, SVG, PDF).
"""

df=sns.load_dataset("penguins")
df.head()

plot=sns.barplot(x="species",y="bill_length_mm",data=df)
plt.title("Average bill length for different species ")
plt.show()

plot.figure.savefig("barplot.png",format="png")
plot.figure.savefig("barplot.svg",format="svg")
plot.figure.savefig("barplot.pdf",format="pdf")

"""## Exporting for Reports : Understand best practices for exporting visualizations for reports or presentations."""

#1. Select the format that best suits your purpose (e.g., PNG for web, SVG for scalable graphics, PDF for documents).

#2. Save your plots at 300 DPI for high-quality prints.

#3. Set the figure size to fit the layout of your report or presentation.

plt.figure(figsize=(12,5))
sns.set_palette("coolwarm")
plot=sns.boxplot(x="species",y="bill_length_mm",hue="sex",data=df)
plt.title("Distribution of bill length for different species for different gender")
plt.legend(loc="best",frameon=False)
plt.ylabel("bill length")
plt.show()

plot.figure.savefig("best_practice_boxplot.png",format="png",dpi=300)
