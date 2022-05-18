
#Base Image to use
FROM python:3.8-slim

#Expose port 8080
EXPOSE 8080

#Optional - install git to fetch packages directly from github
RUN apt-get update && apt-get install -y git

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN python3 -m pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

ENTRYPOINT ["streamlit", "run", "form.py", "--server.port=$PORT", "--server.address=0.0.0.0"]

