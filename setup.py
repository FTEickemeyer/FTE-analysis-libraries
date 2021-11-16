import setuptools

with open( 'README.md', 'r' ) as f:
    long_desc = f.read()

setuptools.setup(
    name='fte-analysis-libraries',
    version = '0.0.1',
    author='Felix Eickemeyer',
    author_email = 'Felix.Eickemeyer@EPFL.ch',
    description = 'An assortment of analysis libraries.',
    long_description = long_desc,
    long_description_content_type = 'text/markdown',
    packages = setuptools.find_packages(),
    package_dir = {'FTE_analysis_libraries': 'FTE_analysis_libraries'},
    package_data = {'FTE_analysis_libraries': ['System_data/*.*', 'System data/Calibration_lamp_spectra/*.*']},
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
