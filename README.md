# LLMS Playground


## Description

Few experiments using Hugging Face models.


## Installation

Clone the repository:

```bash
git clone https://github.com/kobr4/llms-playground.git
cd your-repo
```

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage


For vocal chat bot using Qwen2.5 Instruct model:
```bash
python3 qwen25.py
```

For image generation with an RTX 4070 using quantized Flux model:
```bash
python3 flux-nf4.py
```


## Configuration

Most models require an NVIDIA card with at least 8 GB of VRAM. 

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, feel free to contact:

- GitHub: [@kobr4](https://github.com/kobr4)

