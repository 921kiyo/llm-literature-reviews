import type { NextPage } from "next";
import Head from "next/head";
import Image from "next/image";
import { useRef, useState } from "react";
import { Toaster, toast } from "react-hot-toast";
import DropDown, { VibeType } from "../components/DropDown";
import Footer from "../components/Footer";
import Header from "../components/Header";
import LoadingDots from "../components/LoadingDots";

const Home: NextPage = () => {
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState<String>("");
  const [references, setReferences] = useState([]);
  const [arxivResults, setArxivResults] = useState({});

  const bioRef = useRef<null | HTMLDivElement>(null);

  const [chatLoading, setChatLoading] = useState(false);

  const scrollToBios = () => {
    if (bioRef.current !== null) {
      bioRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  const askQuestion = async (e: any) => {
    e.preventDefault();
    setAnswer("");
    setReferences([]);
    setArxivResults({});
    setLoading(true);
    console.log("asking question");
    const response = await fetch("http://127.0.0.1:8000/search/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        search_term: query,
      }),
    });

    if (!response.ok) {
      throw new Error(response.statusText);
    }
    const data = await response.json();
    setAnswer(data.answer);
    setReferences(data.references);
    setArxivResults(data.arxiv_results);
    setLoading(false);
  };

  const [showChat, setShowChat] = useState<number | null>(null);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatAnswer, setChatAnswer] = useState<String>("");

  const askQuestionToPaper = async (
    e: any,
    url: str,
    parsed_arxiv_results: any
  ) => {
    e.preventDefault();
    console.log("what is parsed_arxiv_results? ");
    console.log(parsed_arxiv_results);
    setChatAnswer("");
    setChatLoading(true);
    const response = await fetch("http://127.0.0.1:8000/chat/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: chatQuestion,
        url: url,
        parsed_arxiv_results: parsed_arxiv_results,
      }),
    });

    if (!response.ok) {
      throw new Error(response.statusText);
    }
    const data = await response.json();
    setChatAnswer(data.answer);
    setChatLoading(false);
  };

  return (
    <div className="flex max-w-5xl mx-auto flex-col items-center justify-center py-2 min-h-screen">
      <Head>
        <title>ArXiv Agent</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Header />
      <main className="flex flex-1 w-full flex-col items-center text-center px-4 mt-12 ">
        <h1 className="sm:text-5xl text-4xl max-w-[708px] font-bold text-slate-900">
          Chat with arXiv papers
        </h1>
        {/* <p className="text-slate-500 mt-5">47,118 bios generated so far.</p> */}
        <div className="max-w-xl w-full">
          <div className="flex mt-10 items-center space-x-3">
            {/* <Image
              src="/search_icon.jpg"
              width={30}
              height={30}
              alt="1 icon"
              className="mb-5 sm:mb-0"
            /> */}
            <p className="text-left font-medium">
              Type your research question{" "}
              {/* <span className="text-slate-500">
                (or write a few sentences about yourself)
              </span>
              . */}
            </p>
          </div>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-black focus:ring-black my-5"
            placeholder={
              "e.g. What is the limitation of the current large language models?"
            }
          />
          {!loading && (
            <button
              className="bg-black rounded-xl text-white font-medium px-4 py-2 hover:bg-black/80 w-full"
              onClick={(e) => askQuestion(e)}
            >
              Search &rarr;
            </button>
          )}
          {loading && (
            <button
              className="bg-black rounded-xl text-white font-medium px-4 py-2 hover:bg-black/80 w-full"
              disabled
            >
              <LoadingDots color="white" style="large" />
            </button>
          )}
        </div>
        <Toaster
          position="top-center"
          reverseOrder={false}
          toastOptions={{ duration: 2000 }}
        />
        <hr className="h-px bg-gray-700 border-1 dark:bg-gray-700" />
        <div className="space-y-10 my-10">
          <div className="space-y-8 flex flex-col items-center justify-center max-w-xl mx-auto">
            {answer ? (
              <div className="font-bold text-xl text-center">
                Answer: <span className="font-normal">{answer}</span>
              </div>
            ) : null}
            {references && references.length > 0 ? (
              <>
                <div className="font-bold text-lg text-center">References:</div>
                {references.map((reference, index) => (
                  <div key={index} className="space-y-2">
                    <a
                      href={reference.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-semibold text-blue-600 hover:text-blue-800"
                    >
                      {reference.title}
                    </a>
                    <div className="italic">{reference.authors}</div>
                    <div className="text-gray-700">{reference.llm_summary}</div>

                    <button
                      onClick={() => {
                        setShowChat(showChat === index ? null : index);
                        setChatQuestion("");
                      }}
                      className="bg-blue-500 rounded-lg text-white font-medium px-2 py-1 hover:bg-blue-700 w-full mt-2"
                    >
                      {showChat === index
                        ? "Close chat"
                        : "Chat with this paper"}
                    </button>

                    {showChat === index && (
                      <div>
                        <textarea
                          value={chatQuestion}
                          onChange={(e) => setChatQuestion(e.target.value)}
                          placeholder="Type your question here..."
                          className="w-full rounded-md border-gray-300 shadow-sm focus:border-black focus:ring-black"
                        />
                        {!chatLoading && (
                          <button
                            onClick={(e) =>
                              askQuestionToPaper(e, reference.url, arxivResults)
                            }
                            className="bg-green-500 rounded-lg text-white font-medium px-2 py-1 hover:bg-green-700 w-full mt-2"
                          >
                            Ask question
                          </button>
                        )}
                        {chatLoading && (
                          <button
                            className="bg-green-500 rounded-lg text-white font-medium px-2 py-1 hover:bg-green-700 w-full mt-2"
                            disabled
                          >
                            <LoadingDots color="white" style="large" />
                          </button>
                        )}

                        {chatAnswer && (
                          <div className="mt-2 p-2 bg-gray-200 rounded-lg">
                            {chatAnswer}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </>
            ) : null}
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Home;
