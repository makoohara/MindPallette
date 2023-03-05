# CP_MVP
MVP of Capstone that takes user's mood and preferred genre as an input and outputs a song recommendation. Works locally.

How to install dependencies: 

`npm install`

How to run the project: 
`npm start`


How it works: 
This code is written in JavaScript and is used to build a web application that communicates with the OpenAI API. The application listens on port 3000 and has two endpoints.

The first endpoint is a GET request that returns the message "Server running on port 3000" when the application is accessed.

The second endpoint is a POST request that receives user input from the body of the request. The user input is used to generate a prompt for the OpenAI API using the OpenAIApi library. The prompt is a request for a song title to listen to based on the user's mood and an optional genre. The API returns a response with a song title and artist name, and the web application generates a second prompt requesting the most listened 30 seconds of the song. The API responds with a response that includes the requested information, and the web application returns the response to the user in a JSON format.

The web application uses the Express.js framework to handle the API requests and the body-parser middleware to parse the JSON data sent by the client. The cors middleware is used to handle cross-origin resource sharing, which allows the application to be accessed by other domains.

The API key used to authenticate with the OpenAI API is stored as a constant variable called OPENAI_API_KEY. The Configuration object is used to configure the API key, and the OpenAIApi object is used to create requests to the OpenAI API.




