import logging
from dotenv import load_dotenv

load_dotenv()

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from prompts import INSTRUCTIONS, WELCOME_MESSAGE
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vectorstore = FAISS.load_local(
    "database/vector_store",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=INSTRUCTIONS,
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        await self.session.say(WELCOME_MESSAGE)

        # 🔍 ทดสอบเรียกฟังก์ชันตรง ๆ
        #answer = await self.ask_about_museum(self.session, "โซน Mixed Reality มีกิจกรรมอะไรบ้าง")
        #await self.session.say("ผลจากการค้นหาเอกสาร: " + answer)

        self.session.generate_reply()

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."
    
    @function_tool
    async def ask_about_museum(self, context: RunContext, question: str) -> str:
        logger.info(f"🟢 called ask_about_museum with: {question}")  # ✅ log เพิ่ม
        """
        ค้นหาคำตอบจากข้อมูลในพิพิธภัณฑ์ เช่น นิทรรศการหรือสิ่งของจัดแสดง
        """
        docs = vectorstore.similarity_search(question, k=3)
        if not docs:
            return "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามนี้"

        # รวมคำตอบจากเอกสาร
        response = "\n".join([doc.page_content.strip() for doc in docs])
        # ย่อให้สั้นลง (optional: สามารถใช้ LLM สรุปอีกทีก็ได้)
        return response[:500] + "..."  # ตัดยาวเกินไป


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info("🚀 Entry point started")
    logger.info(f"🔧 Room name: {ctx.room.name}")

    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=openai.LLM(model="gpt-3.5-turbo-1106", tool_choice="auto"),
        stt=openai.STT(model="whisper-1", language="th"),
        tts=openai.TTS(voice="shimmer"),
        # use LiveKit's turn detection model
        turn_detection=MultilingualModel(),
    )

    logger.info("⚙️ Session created. Preparing to start AgentSession...")

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    logger.info("✅ AgentSession started successfully")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))