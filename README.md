# LLM Itegrated Multimodal Nao Robot

### Clone the repo
```bash
git clone https://github.com/SalehShuvo/Nao-Multimodal.git
```

### Dependencies

```angular2html
conda create -n "Nao" python=3.12
conda activate Nao
pip install -r requirements.txt
```
Place `OPENAI_API KEY` and `TAVILY_API_KEY` in `.env` file.
### Simulate the Robot in Pybullet
```bash
python planner_agent.py
```
Talk to the Nao Robot with long term memory