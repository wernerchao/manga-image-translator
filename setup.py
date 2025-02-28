from setuptools import setup, find_packages

setup(
    name='manga-image-translator',
    version='0.1.0',
    description='A library for translating texts in manga/images.',
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    url='https://github.com/zyddnys/manga-image-translator',  # Replace with the project URL
    packages=["manga_translator"],  # Automatically find packages in the project
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'Pillow',
        'requests',
        'fastapi',
        'uvicorn',
        'colorama',
        'python-dotenv',
        'langdetect',
        'freetype-py',
        'scikit-image',
        'tqdm',
        'cryptography',
        'aiohttp',
        'transformers',
        'sentencepiece',
        'pyclipper',
        'shapely',
        'deepl',
        'httpx',
        'openai',
        'pandas',
        'onnxruntime',
        'timm',
        'py3langid==0.2.2',
        'kornia',
        'bitsandbytes',
        'accelerate',
        'websockets',
        'marshmallow',
        'cython',
        'aioshutil',
        'aiofiles',
        'arabic-reshaper',
        'pyhyphen',
        'langcodes',
        'manga-ocr',
        'pydensecrf @ git+https://github.com/lucasb-eyer/pydensecrf.git',
    ],
    entry_points={
        "console_scripts": [
            "manga_translator=manga_translator.__main__:main",
        ]
    },
    python_requires='>=3.10, <3.12',
)