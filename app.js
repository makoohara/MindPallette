// Importing required modules
const { Configuration, OpenAIApi } = require("openai");
const bodyParser = require("body-parser");
var express = require("express");
const cors = require('cors');

// Defining OpenAI API key
const OPENAI_API_KEY = "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"

// Creating OpenAI API configuration object
const configuration = new Configuration({
  apiKey: OPENAI_API_KEY,
});

// Creating OpenAI API client object
const openai = new OpenAIApi(configuration);

// Creating Express app object
const app = express();
// Defining port for the server to listen on
const port = 3000;

// Enabling CORS middleware
app.use(cors());

// Configuring body parser middleware
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
// Defining a route for the root URL and sending a response
app.get('/', (req, res) => {
  res.send('Server running on port 3000');
});

app.post("/song", (req, res, next) => {
  // Logging the mood sent in the request body
  console.log(req.body.mood)
  // Building the OpenAI API prompt based on the request body
  let prompt = "Give me the title of a song to listen when you are: " + req.body.mood  + " (write just the title)."
  // Adding genre information to the prompt if available
  if(req.body.genre){
    prompt += " Pick from the genre = '" + req.body.genre + "'"
  }
  console.log(prompt)
  // Calling the OpenAI API to generate a song recommendation based on the prompt
  try {
    openai.createCompletion({
      model: "text-davinci-003",
      prompt: prompt,
      temperature: 0.4,
      max_tokens: 64,
      top_p: 1,
      frequency_penalty: 0,
      presence_penalty: 0,
      stop: ["."]
      //request inside request
    }).then((r)=>{ 
      // Building a new OpenAI API prompt based on the song recommendation
      let prompt2 = "What is the most listened 30 seconds of the song" + r.data.choices[0].text + "? (give me the title and artist, start and end as response)"
      console.log(prompt2)

      // Calling the OpenAI API to generate a response based on the new prompt
      openai.createCompletion({
        model: "text-davinci-003",
        prompt: prompt2,
        temperature: 0.4,
        max_tokens: 64,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
      }).then((r2)=>{ 
        // Logging the response generated by the second OpenAI API call
        console.log(r2.data.choices[0].text)
        // Sending the response back to the client
        res.json({response: r2.data.choices[0].text})
    
      })
      
  
    })
    
  } catch (error) {
    // Handling errors by sending an error response
    res.send(error)
    
  }
  
 });
// Starting the server and logging a message to the console
app.listen(port, () => console.log(`Hello world app listening on port ${port}!`));