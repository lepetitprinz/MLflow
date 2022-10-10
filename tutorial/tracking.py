import os
from random import random, randint

from mlflow import log_metric, log_param, log_artifcats


if __name__ == "__main__":
	print("RUniing tracking.py")

	log_param("param1", randint(0, 100))

	log_metric("foo", random())
	log_metric("foo", random() + 1)
	log_metric("foo", random() + 2)

	if not os.path.exists("outputs"):
		os.makedir("outputs")
	with open("outputs/test.text", "w") as file:
		file.write("it is test")

	log_artifcats("outputs")