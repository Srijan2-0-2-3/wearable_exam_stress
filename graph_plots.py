import pickle
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
datas = pickle.load(open('replay_wearable_exam_stress.pkl', 'rb'), encoding='latin1')
print(datas)


def traverse_dict_and_lists(data, indent=""):
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}{key}:")
            traverse_dict_and_lists(value, indent + "  ")
    elif isinstance(data, list):
        for item in data:
            print(f"{indent}- (List Item):")
            traverse_dict_and_lists(item, indent + "  ")
    else:
        print(f"{indent}{data}")


traverse_dict_and_lists(datas)
experience = len()
# epoch = [0]
# # Initialize empty lists to store metrics
# strategies = []
# final_losses = []
# final_accuracies = []
#
# # Iterate through the data to extract relevant information
# for item in datas:
#     if 'strategy' in item:
#         strategies.append(item['strategy'])
#         final_losses.append(item['finalloss'])
#         final_accuracies.append(item['finalacc'])
#
# # Create a bar chart for final loss and accuracy
# plt.figure(figsize=(10, 6))
# plt.bar(strategies, final_losses, label='Final Loss', color='blue')
# plt.bar(strategies, final_accuracies, label='Final Accuracy', color='green', alpha=0.5)
#
# # Set labels and title
# plt.xlabel('Strategy')
# plt.ylabel('Value')
# plt.title('Final Loss and Accuracy by Strategy')
# plt.legend()
#
# # Show the plot
# plt.grid()
# plt.show()
# plt.savefig('plot.png')