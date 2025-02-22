type AstroType = {
  url: URL;
};
export const resolve = (astro: AstroType, url: string) => {
  return new URL(url, astro.url);
};
