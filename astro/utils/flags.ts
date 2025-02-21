import fs from "node:fs/promises";
import { type FlagModelMulti, type FlagModelOne } from "./types";

const getQrCode = async (path: string) => {
  const files = await fs.readdir(`flags/${path}`);
  return files.filter((file) => /^\d+\.png$/.test(file)).map((file) => `/flags/${path}/${file}`);
};

export const getQrOne = async (path: string) => {
  const json = await fs.readFile(`flags/${path}/data.json`, "utf-8");
  const data = JSON.parse(json) as FlagModelOne[];
  const flag = await Promise.all(data.map(async (f) => ({ ...f, qrcode: await getQrCode(`${path}/${f.name}`) })));
  return flag;
};

export const getQrMulti = async (path: string) => {
  const json = await fs.readFile(`flags/${path}/data.json`, "utf-8");
  const data = JSON.parse(json) as FlagModelMulti[];
  const flag = await Promise.all(data.map(async (f) => ({ ...f, qrcode: await getQrCode(`${path}/${f.name}`) })));
  return flag;
};
