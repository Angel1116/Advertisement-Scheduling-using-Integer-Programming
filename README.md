# Advertisement Scheduling using Integer Programming

## Introduction
In this project, we describe an advertisement scheduling system developed for a broadcast company providing a platform for streaming anime advertisements. The
company needs to decide which ads to send out to which users at what particular time, given a limited capacity of broadcast time slots, while maximizing user response and revenues from anime advertisers paying for each ad. We formulate it as an optimization problem and use integer programming to solve it. Additionally, we use K-Nearest Neighbor (KNN) to predict the preferred ad schedule for users who have not rated enough ads. We show that the ad scheduling system significantly reduces the time by automating the ad scheduling, and predicts the user’s preferred ad scheduling by the other users having the most similar preference.

## Programming structure chart
After downloding all the datasets and codes, you can run "optimization.py" to optimize the ad scheduling. The programming structure chart shows as follows:
<img src="https://github.com/Angel1116/Integer-Programming-for-Advertisement-Scheduling/assets/103301338/fc033511-4a9a-4426-96e9-369748ac8688" width="600"/>

You can follow the command and select your prompt. There are two choices shown as below:

## Choice 1: schedule for a user in database
If you choose the first choice, you would input as follows:
<img src="https://github.com/Angel1116/Integer-Programming-for-Advertisement-Scheduling/assets/103301338/d012768c-f12a-469c-9f4d-626926290262" width="550"/>

It would output the schedule in Excel.

If you don't have enough time to execute the code, you could download example "schedule_1.xlsx" to check the output schedule.

<img src="https://github.com/user-attachments/assets/2a4582f7-fbee-4a11-ad06-33f9ab0a2bab" width="550"/>

▲Scheduling result with ad index

<img src="https://github.com/user-attachments/assets/6c3930cb-6a4d-4454-a999-3a6e9b7441ed" width="550"/>

▲Scheduling result with ad genres

## Choice 2: upload a new data and schedule for a new user
If you choose the second choice, you could choose "sample_A_df.csv" or "sample_B_df.csv" or "sample_C_df.csv" to upload and input as follows:

<img src="https://github.com/Angel1116/Integer-Programming-for-Advertisement-Scheduling/assets/103301338/8502f19d-cb3a-4e50-a269-43afd85ac0f4" width="550"/>

After running the second choice, it would output the schedule in Excel and a bar chart about Jaccard similarity as follows:
<img src="https://github.com/Angel1116/Integer-Programming-for-Advertisement-Scheduling/assets/103301338/ef483bf4-7c46-4e36-8136-f69532ef5887" width="550"/>

▲Jaccard similarity comparison for ads and ad genres
If you don't have enough time to execute the code, you could download example "schedule_1.xlsx" to check the output schedule.
