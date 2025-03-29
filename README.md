How to Run the Application
This project is designed to group similar HTML pages based on their visual appearance by comparing full-page screenshots (using SSIM).

Prerequisites
Python 3.8+ is required.

It's recommended to use a virtual environment.

Installation
Clone the repository:

Copy: git clone https://github.com/Sala-Sergiu/HTML-CLONES_Veridion_Project.git
cd HTML-CLONES_Veridion_Project
Create and activate a virtual environment:

On Linux/Mac:

Copy: python3 -m venv venv
source venv/bin/activate
On Windows:

Copy: python -m venv venv
venv\Scripts\activate
Install dependencies:

Copy: pip install -r requirements.txt
Running the Application
The project contains two main modules:

visual_similarity.py: Contains functions for loading HTML pages, taking screenshots, and calculating the SSIM between images.

group_visual.py: Uses functions from visual_similarity.py to process a directory of HTML files, group similar pages based on SSIM, and organize the output.

To run the grouping task:
Make sure you have your HTML files organized in a folder (e.g., clones/tier3). Then run:

Copy: python group_visual.py --input_dir clones/tier3 --screenshot_dir screenshots --output_json output/tier3.json --output_csv output/tier3.csv --error_output output/errors.json --image_threshold 0.3 --port 8000 --max_workers_features 4 --max_workers_pairs 4 --use_indexing
Parameters:

--input_dir: The directory containing the HTML files (e.g., clones/tier3).
--screenshot_dir: Directory where screenshots will be saved.
--output_json and --output_csv: Paths for the output files listing the groups.
--error_output: Path for the JSON file logging errors.
--image_threshold: The SSIM threshold; pages with SSIM above this value are considered similar.
--port: The port for the local HTTP server used to serve HTML files.
--max_workers_features and --max_workers_pairs: Control the parallelism for processing files and comparing pairs.
--use_indexing: If enabled, reduces the number of comparisons by grouping files based on a signature.

To test similarity between two HTML files:
You can use the testing mode in visual_similarity.py directly. For example:

Copy: python visual_similarity.py --file1 path/to/first.html --file2 path/to/second.html --screenshot_dir screenshots --port 8000
This will output the CSS-based and SSIM-based similarity scores and save the screenshots.

Additional Notes
Local HTTP Server: The application starts a local HTTP server to serve HTML files. Ensure that your HTML files and all resources are accessible.

Dynamic Content: If your pages load content dynamically (via JavaScript), the script waits for the network to become idle and for all images to load before taking screenshots.

Output Organization: Screenshots are organized by tier (e.g., screenshots/tier3) and further grouped by similarity in subdirectories like group_1, group_2, etc.

