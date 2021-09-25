import numpy as np
import pandas as pd
import praw
import nltk
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


def scrape_reddit(limit):

    '''
    This function uses the praw library to scrape the 'peloton' subreddit.
    Returns a dataframe with each row representing a comment and information on the post.

    args:
    limit:number of posts to select (integer).
    '''

    SECRET = '' # deleted for privacy
    ID = ''# deleted for privacy

    USER_AGENT = 'Scraper 1.0'
    reddit = praw.Reddit(
        client_id = ID,
        client_secret = SECRET,
        user_agent = USER_AGENT
    )

    storage_list = []
    for submission in reddit.subreddit('peloton').hot(limit=limit):
        title = submission.title
        if not 'RFL 21' in title: # remove competition posts
            id_ = submission.id
            author = submission.author
            time = submission.created_utc
            score = submission.score
            ratio = submission.upvote_ratio


            submission.comments.replace_more(limit=0)


            for comment in submission.comments.list():
                comment_dict = {'post_id': id_,
                                'post_title':title,
                               'post_author': author,
                               'post_time': time,
                               'post_score': score,
                                'post_ratio': ratio,
                                # comments
                                'comment_id' : comment.id,
                               'comment_author': comment.author,
                                'comment_body': comment.body.lower(),
                                'comment_score' : comment.score,
                                'comment_contriv': comment.controversiality,
                                'parent_id': comment.parent()}
                storage_list.append(comment_dict)

    df=pd.DataFrame(storage_list)

    return df



def get_rider_sentiment(df, names):

    '''
    This function will perform sentiment analysis on each comment mentioning the names entered

    args:
    df: scraped data, output of scrape_reddit
    names: a list of names the rider is known by. This could include nicknames and abbreviations.

    '''

    names = [n.lower() for n in names] # make sure all are lower
    rider_data = df[df['comment_body'].str.contains(r'\b(?:{})\b'.format('|'.join(names)))] # find mentions

    sia = SIA()
    result_store = []

    # sentiment analyser
    for text in rider_data.comment_body:
        pol_score = sia.polarity_scores(text)
        pol_score['text'] = text
        result_store.append(pol_score)

    pol_df = pd.DataFrame(result_store)

    # now create a label
    pol_df['label'] = 'Neutral'
    pol_df.loc[pol_df['compound'] > 0.2, 'label'] = 'Positive' # positive
    pol_df.loc[pol_df['compound']<-0.2, 'label'] = 'Negative' # negative

    # counts
    cnts = pol_df['label'].value_counts(normalize = True)*100
    p=cnts.plot(kind='bar', ylabel = 'Percentage', figsize = (12,8))

    print(f'A total of {len(rider_data)} comments mentioned this rider')
    plt.show();
