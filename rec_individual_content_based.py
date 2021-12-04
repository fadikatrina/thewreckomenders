from logger import l
import fasttext
from helpers_general import clean_text, get_recipe_details
import settings


def fasttext_keywords():
	l.info("Recommendation using fasttext keywords started")
	model = fasttext.load_model(settings.FASTTEXT_MODEL_PATH_QUANTIZED)
	while True:
		user_input = input("Enter your query, or 1 to quit ")
		if user_input == "1":
			break
		l.info(f"User input: {user_input}")
		clean_user_input = clean_text(user_input)
		result = model.predict(clean_user_input, k=5)
		result = extract_ft_rec(result)
		l.info(f"Prediction {result}")


def extract_ft_rec(rec):
	rec_labels = rec[0]
	recipe_details = []
	for rec_label in rec_labels:
		rec_label = rec_label.split("_")
		print(rec_label)
		label_type = rec_label[-2]
		print(label_type)
		label_value = rec_label[-1]
		print(label_value)
		if label_type in ["recipe"]:
			print("true")
			recipe_details.append(get_recipe_details(label_value))
	return recipe_details


if __name__ == "__main__":
	fasttext_keywords()
