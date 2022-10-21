# 동시성 프로그래밍
import aiohttp
import asyncio
import time

async def fetcher(session, url):
    async with session.get(url) as resp:
        return await resp.text()

async def main():
    urls = ['https://naver.com', 'https://google.com', 'https://instagram.com']

    async with aiohttp.ClientSession() as s:
        result = await asyncio.gather(*[fetcher(s, url) for url in urls])
        print(result)


if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    print(time.time() - start)

