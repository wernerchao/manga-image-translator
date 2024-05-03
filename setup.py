from setuptools import setup

setup(
    name="manga-image-translator",
    version="0.0.2",
    description="Some manga/images will never be translated, therefore this project is born.",
    url="https://github.com/zyddnys/manga-image-translator",
    author="zyddnys",
    author_email="",
    license="GNU General Public License v3.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=["manga_translator"],
    include_package_data=True,
    install_requires=[
        "networkx",
        "torch",
        "torchvision",
        "torch-summary",
        "einops",
        "scikit-image",
        "opencv-python",
        "pyclipper",
        "shapely",
        "requests",
        "cryptography",
        "freetype-py",
        "googletrans==4.0.0rc1",
        "aiohttp",
        "tqdm",
        "deepl",
        "ImageHash",
        "kornia",
        "backports.cached-property",
        "huggingface_hub",
        "transformers",
        "py3langid",
        "sentencepiece",
        "editdistance",
        "numpy",
        "tensorboardX",
        "websockets",
        "protobuf",
        "ctranslate2",
        "colorama",
        "openai==0.28",
        "open_clip_torch",
        "safetensors",
        "pandas",
        "onnxruntime",
        "timm",
        "omegaconf",
        "python-dotenv",
        "nest-asyncio",
        "marshmallow",
        "cython",
        "aioshutil",
        "aiofiles",
        "arabic-reshaper",
        "pyhyphen",
        "langcodes",
        "manga-ocr",
    ],
    entry_points={
        "console_scripts": [
            "manga_translator=manga_translator.__main__:main",
        ]
    },
)
