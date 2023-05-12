import Document, { Head, Html, Main, NextScript } from "next/document";

class MyDocument extends Document {
  render() {
    return (
      <Html lang="en">
        <Head>
          <link rel="icon" href="/favicon.ico" />
          <meta name="description" content="Chat with arXiv papers." />
          <meta property="og:site_name" content="ResearchAgent.com" />
          <meta property="og:description" content="Chat with arXiv papers." />
          <meta property="og:title" content="ArXiv Agent" />
          <meta name="twitter:card" content="summary_large_image" />
          <meta name="twitter:title" content="ArXiv Agent" />
          <meta name="twitter:description" content="Chat with arXiv papers." />
          <meta
            property="og:image"
            content="https://ResearchAgent.com/og-image.png"
          />
          <meta
            name="twitter:image"
            content="https://ResearchAgent.com/og-image.png"
          />
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
