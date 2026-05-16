export default {
  testEnvironment: 'node',
  extensionsToTreatAsEsm: ['.ts'],
  transform: {
    '^.+\\.ts$': ['ts-jest', {
      useESM: true
    }]
  },
  moduleFileExtensions: ['ts', 'js', 'mjs'],
  testMatch: ['**/*.test.ts'],
  testPathIgnorePatterns: [],
  verbose: true,
  transformIgnorePatterns: [],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1'
  }
};