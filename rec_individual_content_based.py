from logger import l
import fasttext
from helpers_general import clean_text
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
		l.info(f"Prediction {result}")
