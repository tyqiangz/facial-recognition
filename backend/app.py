from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Face Similarity",
              description="""Given two images, produce a similarity score based on how similar the faces are.
              """)

# mount the static files for use by the webpage
app.mount("/static",
          StaticFiles(directory="static"), name="static")

# jinja templates
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def startup_event():
    ''''
    Load some objects upon start-up to be used by the APIs
    '''
    pass

@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/normalise-face')
def normalise_face(face = {"hello": "world"}):
    '''
    Aligns a face
    '''

    return face

if __name__ == "__main__":
    # for running in HTTP
    uvicorn.run("app:app", reload=False, debug=True, host="0.0.0.0", port=8555)
    # for running in HTTPS
    # uvicorn.run(app, host="0.0.0.0", port=8555, reload=False, debug=True, 
    #             proxy_headers=True, forwarded_allow_ips="*")