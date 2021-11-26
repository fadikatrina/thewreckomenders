from logger import l
import fasttext
import settings
import errno


def fasttext_auto_eval():
	l.info("Auto evaluation of fasttext model using testing file")
	l.info("Testing normal model")
	model = fasttext.load_model(settings.FASTTEXT_MODEL_PATH_NORMAL)
	log_results(*model.test("Data/processed/fasttext_testing_enriched.txt"))
	l.info("Testing quantized model")
	model = fasttext.load_model(settings.FASTTEXT_MODEL_PATH_QUANTIZED)
	log_results(*model.test("Data/processed/fasttext_testing_enriched.txt"))


def log_results(N, p, r):
	l.info("N\t" + str(N))
	l.info("P@{}\t{:.3f}".format(1, p))
	l.info("R@{}\t{:.3f}".format(1, r))


def show_words_labels():
	l.info("Fasttext logging words/labels and frequencies")
	model = fasttext.load_model(settings.FASTTEXT_MODEL_PATH_NORMAL)
	labels, freq = model.get_labels(include_freq=True)
	l.info("Fasttext Labels:")
	log_words_labels(labels, freq)
	words, freq = model.get_words(include_freq=True)
	l.info("Fasttext Words:")
	log_words_labels(words, freq)


def log_words_labels(w_or_l, freq):
	for w, f in zip(w_or_l, freq):
		try:
			l.info(w + "\t" + str(f))
		except IOError as e:
			if e.errno == errno.EPIPE:
				pass


if __name__ == "__main__":
	show_words_labels()
	fasttext_auto_eval()
