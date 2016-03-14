from dateutil.parser import parse
import numpy as np
import pandas as pd
import pylab

def get_consumed_days(day_1, day2):
    diff = day2 - day_1
    return diff.days + 1

start_date = parse("14 Oct 2015")
end_date = parse("11 Dec 2015")
min_consumed_days = 1
max_consumed_days = get_consumed_days(start_date, end_date)

TeamName_list = ["nobody", "陈 日伟", "TUT_Newbie(256481,256311,256275)"]
DisplayName_list = ["Team nobody", "Team chenriwei", "Team newbie"]
color_list = ["yellowgreen", "lightskyblue", "darkorange"]

# Read file content
file_content = pd.read_csv("./AllSubmissionsDetails.csv")

pylab.figure()
for TeamName, DisplayName, color in zip(TeamName_list, DisplayName_list, color_list):
    # Retrieve records
    selected_indexes = np.array(file_content[" TeamName"] == TeamName)
    selected_records = file_content.as_matrix([" DateSubmittedUtc", " PrivateScore"])[selected_indexes, :]

    # Only keep the record which improves the score
    current_best_score = -np.inf
    improving_score_index_flag = np.zeros(selected_records.shape[0], dtype=bool)
    for current_record_index in np.arange(selected_records.shape[0]):
        if selected_records[current_record_index, 1] > current_best_score:
            improving_score_index_flag[current_record_index] = True
            current_best_score = selected_records[current_record_index, 1]
    selected_records = selected_records[improving_score_index_flag, :]

    # Calculate the consumed days
    selected_records[:, 0] = [get_consumed_days(start_date, parse(current_record[0])) for current_record in selected_records]

    # Only keep the best score in the same day
    best_score_index_flag = np.zeros(selected_records.shape[0], dtype=bool)
    for current_record_index in np.flipud(np.arange(selected_records.shape[0])):
        if selected_records[current_record_index, 0] not in selected_records[best_score_index_flag, 0]:
            best_score_index_flag[current_record_index] = True
    selected_records = selected_records[best_score_index_flag, :]

    # Insert the score on the last day if necessarily
    if max_consumed_days != selected_records[-1, 0]:
        selected_records = np.vstack([selected_records, [max_consumed_days, selected_records[-1, 1]]])

    pylab.step(selected_records[:, 0], selected_records[:, 1], where="post", color=color, label=DisplayName)

pylab.xlabel("Elapsed Days")
pylab.ylabel("Weighted AUC Score on the Private Leaderboard")
pylab.grid()
pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
pylab.show()
