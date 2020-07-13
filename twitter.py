import tweepy

class Twitter :
    def __init__(self):
        print("hehe")
        pass

    def instance(self):
        CONSUMER_KEY = "WlY1hh8ipTdWgJSUlI6pyfzO3"
        CONSUMER_SECRET = "r3MTOBbLzPsFtn4vt2gZfCaA37G206ODLhsEihCA0OJpSokn6z"
        ACCESS_KEY = "356629975-7WECD0yqcRA4YDP4EEMkIxIvK8833Gnoa9Ol1uoT"
        ACCESS_SECRET = "G8uWeSZSsGhfBIC4ACwIclasAE00lDHAIkhHV24mK1mRm"
        api = tweepy.OAuthHandler(consumer_key = CONSUMER_KEY, consumer_secret = CONSUMER_SECRET)
        api.set_access_token(ACCESS_KEY, ACCESS_SECRET)
        return tweepy.API(api, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
