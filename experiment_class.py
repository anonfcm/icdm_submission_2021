from fuzzy import *
import time
import pickle
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.cluster import KMeans

class Runtimes():
	def __init__(self, results_filename):
		self.results_filename = results_filename + "runtimes.txt"
		self.full_times = []
		self.mmdrs_times = []
		self.random_vw_coreset_times = []
		self.random_sample_times = []
		self.km_times = []
		self.rsekm_times = []
		self.mmdrs_km_times = []
		
	def write_individual_result(self, f, relative_results, algo):
		f.write(algo + ", " + str(np.mean(relative_results)) + ", " + str(np.std(relative_results)) + "\n")
	
	def save_results(self):
		f = open(self.results_filename, "w")
		print("DEBUG: saving runtime results to:", self.results_filename)
		self.write_individual_result(f, self.full_times, "full")
		self.write_individual_result(f, self.mmdrs_times, "mmdrs")
		self.write_individual_result(f, self.random_vw_coreset_times, "random_vw_coreset")
		self.write_individual_result(f, self.random_sample_times, "random_sample")
		self.write_individual_result(f, self.km_times, "km")
		self.write_individual_result(f, self.rsekm_times, "rsekm")
		self.write_individual_result(f, self.mmdrs_km_times, "mmdrs_km")
		f.close()
		return

class Errors():
	def __init__(self, results_filename):
		self.results_filename = results_filename + "errors.txt"
		self.full_errors = []
		self.mmdrs_errors = []
		self.random_vw_coreset_errors = []
		self.random_sample_errors = []
		self.km_errors = []
		self.rsekm_errors = []
		self.mmdrs_km_errors = []
		

	def get_relative_errors(self):
		self.full_relative = self.calculate_relative_errors(self.full_errors, 
			self.full_errors)
		self.mmdrs_relative = self.calculate_relative_errors(self.mmdrs_errors, 
			self.full_errors)
		self.random_vw_coreset_relative = self.calculate_relative_errors(self.random_vw_coreset_errors, 
			self.full_errors)
		self.random_sample_relative = self.calculate_relative_errors(self.random_sample_errors, 
			self.full_errors)
		self.km_relative = self.calculate_relative_errors(self.km_errors, 
			self.full_errors)
		self.rsekm_relative = self.calculate_relative_errors(self.rsekm_errors, 
			self.full_errors)
		self.mmdrs_km_relative = self.calculate_relative_errors(self.mmdrs_km_errors, 
			self.full_errors)	
		
	def calculate_relative_errors(self, coreset_errors, full_errors):
		x = np.absolute(np.divide(np.subtract(coreset_errors, full_errors), full_errors))
		return x * 100

	def write_individual_result(self, f, relative_results, algo):
		f.write(algo + ", " + str(np.mean(relative_results)) + ", " + str(np.std(relative_results)) + "\n")
	
	def save_results(self):
		f = open(self.results_filename, "w")
		print("DEBUG: saving errors results to:", self.results_filename)
		self.write_individual_result(f, self.full_relative, "full")
		self.write_individual_result(f, self.mmdrs_relative, "mmdrs")
		self.write_individual_result(f, self.random_vw_coreset_relative, "random_vw_coreset")
		self.write_individual_result(f, self.random_sample_relative, "random_sample")
		self.write_individual_result(f, self.km_relative, "km")
		self.write_individual_result(f, self.rsekm_relative, "rse_km")
		self.write_individual_result(f, self.mmdrs_km_relative, "mmdrs_km")
		f.close()
		return

class Experiment():
	def __init__(self, results_filename, dataset, k, coreset_size, num_loops):
		self.results_filename = results_filename

		# automatically scales the dataset.
		scaler = MinMaxScaler()
		self.dataset = scaler.fit_transform(dataset)
		
		self.k = k
		self.coreset_size = coreset_size
		self.num_loops = num_loops
		return

	def run_experiment(self):
		
		errors = Errors(self.results_filename)
		runtimes = Runtimes(self.results_filename)
		for i in range(self.num_loops):
			loop_start = time.time()
			#random data 

			# run the full FCM
			start = time.time()
			centroids, init_centroids, u, init_u, d, t = fcm(self.dataset, self.k)
			end = time.time()
			curr_error = fcm_error(self.dataset, centroids)
			runtimes.full_times.append(end - start)
			errors.full_errors.append(curr_error)

			# run the with mmdrs
			start = time.time()
			mmdrs_data = mmdrs(self.dataset, self.k + 1, self.coreset_size)
			centroids, init_centroids, u, init_u, d, t = fcm(mmdrs_data, self.k)
			end = time.time()
			curr_error = fcm_error(self.dataset, centroids)
			runtimes.mmdrs_times.append(end - start)
			errors.mmdrs_errors.append(curr_error)

			# run with random Voronoi weighted coreset.
			start = time.time()
			random_data = random_sample(self.dataset, self.coreset_size)
			weights = get_voronoi_weights(self.dataset, random_data)
			centroids, init_centroids, u, init_u, d, t = fcm(random_data, self.k, weights = weights)
			end = time.time()
			curr_error = fcm_error(self.dataset, centroids)
			runtimes.random_vw_coreset_times.append(end - start)
			errors.random_vw_coreset_errors.append(curr_error)

			# run the rseFCM
			start = time.time()
			centroids, init_centroids, u, init_u, d, t = fcm(random_data, self.k)
			end = time.time()
			curr_error = fcm_error(self.dataset, centroids)
			runtimes.random_sample_times.append(end - start)
			errors.random_sample_errors.append(curr_error)

			# run with k-means++
			start = time.time()
			kmeans = KMeans(n_clusters = self.k, init = d2_seeding(self.dataset, self.k))
			kmeans.fit(random_data)
			centroids = kmeans.cluster_centers_
			end = time.time()
			curr_error = fcm_error(self.dataset, centroids)
			runtimes.km_times.append(end - start)
			errors.km_errors.append(curr_error)


			# run with rse-k-means++
			start = time.time()
			random_data = random_sample(self.dataset, self.coreset_size)
			kmeans = KMeans(n_clusters = self.k, init = d2_seeding(random_data, self.k))
			kmeans.fit(random_data)
			centroids = kmeans.cluster_centers_
			end = time.time()
			curr_error = fcm_error(self.dataset, centroids)
			runtimes.rsekm_times.append(end - start)
			errors.rsekm_errors.append(curr_error)

			# run with MMDRS-k-means++
			start = time.time()
			mmdrs_data = mmdrs(self.dataset, self.k + 1, self.coreset_size)
			kmeans = KMeans(n_clusters = self.k, init = d2_seeding(mmdrs_data, self.k))
			kmeans.fit(mmdrs_data)
			centroids = kmeans.cluster_centers_
			end = time.time()
			curr_error = fcm_error(self.dataset, centroids)
			runtimes.mmdrs_km_times.append(end - start)
			errors.mmdrs_km_errors.append(curr_error)

			print("loop", i, "took", time.time() - loop_start, "seconds!")

		# save the errors and runtime objects.
		runtimes.save_results()
		errors.get_relative_errors()
		errors.save_results()

		pickle.dump(runtimes, open(self.results_filename + "_runtimes", "wb"))
		pickle.dump(errors, open(self.results_filename + "_errors", "wb"))
		return
