import Document, { Head, Html, Main, NextScript } from "next/document";

class MyDocument extends Document {
  render() {
    return (
      <Html lang="en">
        <Head>
          <link rel="icon" href="/favicon.ico" />
          <meta
            name="description"
            content="Get insights from research papers in seconds."
          />
          <meta property="og:site_name" content="ResearchAgent.com" />
          <meta
            property="og:description"
            content="Get insights from research papers in seconds."
          />
          <meta property="og:title" content="Research Agent" />
          <meta name="twitter:card" content="summary_large_image" />
          <meta name="twitter:title" content="Research Agent" />
          <meta
            name="twitter:description"
            content="Get insights from research papers in seconds."
          />
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