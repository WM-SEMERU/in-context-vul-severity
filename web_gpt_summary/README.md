# cisco-bugs-vulnerabilities
cisco-bugs-vulnerabilities

# prepare

## prepare dataset
Unzip file

```
python -m zipfile -e ./data/cvelistV5-main.zip ./Data/
```

## setup openai key

```
export OPENAI_API_KEY='your-api-key-here'
```

# Run

```
python main_web_crawler.py
```


# Proj Structure

```
├── cache: temporary file during execution
├── data: raw json data
├── util: utility python module
```


# TODO List

- [ ] Document for analyzing features or required inputs/outs for our model. 
- [ ] Wondering how to implement the codegen with the in-context learning using CISCO dataset 
- [ ] Which model is the most intuitive for the in-context learning?(GPT2 related models)
- [ ] [@Anna Schmedding] Dataset statistical analysis
- [ ] Collect and organize datasets from external sources
- [ ] [@Daniel] Implement the data extraction for github
- [ ] Implement data extraction from twitter
- [ ] [@Yiyang Lu] Following the references we can apply web scrapping and use ChatGPT to summarize the content
- [ ] [@Daniel] Look for paper references about how to extract data from reported CVE vulnerabilities, How to summarize web pages? Justify the use of ChatGPT
